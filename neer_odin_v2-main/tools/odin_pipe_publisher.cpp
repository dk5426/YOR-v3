#include "lidar_api.h"
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#include <cstring>
#include <atomic>
#include <csignal>
#include <mutex>
#include <vector>

device_handle odinDevice = nullptr;
bool deviceConnected = false;
int pipe_fd = -1;
std::atomic<bool> keep_running(true);
std::mutex pipe_mutex;

// Buffers for background writing
struct FrameData {
    uint32_t type;
    uint32_t arg1;
    uint32_t arg2;
    std::vector<uint8_t> payload;
};
std::mutex queue_mutex;
std::vector<FrameData> frame_queue;
const size_t MAX_QUEUE_SIZE = 2; // Drop old frames if Python is too slow

const char* PIPE_NAME = "/tmp/odin_rgb_pipe";

const uint32_t MAGIC_IMAGE = 0x11223344;
const uint32_t MAGIC_POINTCLOUD = 0x55667788;

void signal_handler(int sig) {
    std::cout << "Interrupt received, stopping..." << std::endl;
    keep_running = false;
}

void pipe_writer_thread() {
    while (keep_running) {
        FrameData frame;
        bool has_frame = false;
        
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!frame_queue.empty()) {
                frame = std::move(frame_queue.front());
                frame_queue.erase(frame_queue.begin());
                has_frame = true;
            }
        }
        
        if (has_frame && pipe_fd > 0) {
            uint32_t header[4] = {frame.type, frame.arg1, frame.arg2, (uint32_t)frame.payload.size()};
            std::lock_guard<std::mutex> lock(pipe_mutex);
            ssize_t written = write(pipe_fd, header, sizeof(header));
            if (written == sizeof(header)) {
                // Write payload in chunks to avoid overwhelming pipe?
                // Just a single blocking write is fine since we are in a background thread
                write(pipe_fd, frame.payload.data(), frame.payload.size());
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

static void lidar_data_callback(const lidar_data_t *data, void *user_data) {
    if (!deviceConnected || !keep_running || pipe_fd <= 0) return;

    if (data->type == LIDAR_DT_RAW_RGB) {
        if (data->stream.imageCount > 0) {
            auto img = data->stream.imageList[0];
            if (img.pAddr != nullptr && img.length > 0) {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (frame_queue.size() < MAX_QUEUE_SIZE) {
                    FrameData frame;
                    frame.type = MAGIC_IMAGE;
                    frame.arg1 = img.width;
                    frame.arg2 = img.height;
                    frame.payload.assign((uint8_t*)img.pAddr, (uint8_t*)img.pAddr + img.length);
                    frame_queue.push_back(std::move(frame));
                }
            }
        }
    } 
    else if (data->type == LIDAR_DT_SLAM_CLOUD) {
        if (data->stream.imageCount >= 1) {
            auto pc = data->stream.imageList[0];
            if (pc.pAddr != nullptr && pc.length > 0) {
                uint32_t num_points = pc.length / (7 * sizeof(int32_t));
                
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (frame_queue.size() < MAX_QUEUE_SIZE) {
                    FrameData frame;
                    frame.type = MAGIC_POINTCLOUD;
                    frame.arg1 = num_points;
                    frame.arg2 = 0;
                    frame.payload.assign((uint8_t*)pc.pAddr, (uint8_t*)pc.pAddr + pc.length);
                    frame_queue.push_back(std::move(frame));
                }
            }
        }
    }
}

static void lidar_device_callback(const lidar_device_info_t* device_info, bool attach) {
    if (attach) {
        std::cout << "Hardware connected. Initializing device..." << std::endl;
        if (lidar_create_device((lidar_device_info_t*)device_info, &odinDevice) != 0) {
            std::cerr << "Failed to create device" << std::endl;
            return;
        }

        lidar_data_callback_info_t cb_info;
        cb_info.data_callback = lidar_data_callback;
        cb_info.user_data = nullptr;

        if (lidar_register_stream_callback(odinDevice, cb_info) != 0) {
            std::cerr << "Register callback failed" << std::endl;
            return;
        }

        if (lidar_open_device(odinDevice) != 0) {
            std::cerr << "Failed to open device" << std::endl;
            return;
        }

        lidar_set_mode(odinDevice, LIDAR_MODE_SLAM);
        
        // Activate streams required for SLAM and our output
        lidar_activate_stream_type(odinDevice, LIDAR_DT_RAW_RGB);
        lidar_activate_stream_type(odinDevice, LIDAR_DT_RAW_IMU);
        lidar_activate_stream_type(odinDevice, LIDAR_DT_RAW_DTOF);
        lidar_activate_stream_type(odinDevice, LIDAR_DT_SLAM_CLOUD);
        lidar_activate_stream_type(odinDevice, LIDAR_DT_SLAM_ODOMETRY);
        
        uint32_t odr = 0;
        // Start all streams for the selected mode (LIDAR_MODE_SLAM)
        if (lidar_start_stream(odinDevice, LIDAR_MODE_SLAM, odr) != 0) {
            std::cerr << "Start stream failed" << std::endl;
        }

        deviceConnected = true;
        std::cout << "Device ready and streaming!" << std::endl;
    } else {
        std::cout << "Device disconnected." << std::endl;
        deviceConnected = false;
        keep_running = false;
    }
}

int main() {
    std::signal(SIGINT, signal_handler);
    std::signal(SIGPIPE, SIG_IGN); // Ignore broken pipe to prevent exit/segfault
    
    mkfifo(PIPE_NAME, 0666);
    
    std::cout << "Waiting for Python subscriber to open pipe " << PIPE_NAME << "..." << std::endl;
    pipe_fd = open(PIPE_NAME, O_WRONLY);
    if (pipe_fd < 0) {
        std::cerr << "Failed to open pipe for writing" << std::endl;
        return -1;
    }
    
    std::cout << "Pipe opened. Initializing Lidar SDK..." << std::endl;

    std::thread writer(pipe_writer_thread);

    if (lidar_system_init(lidar_device_callback) != 0) {
        std::cerr << "Lidar system init failed" << std::endl;
        return -1;
    }

    while (keep_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (deviceConnected && odinDevice) {
        // Detach streams gracefully before shutdown
        lidar_stop_stream(odinDevice, LIDAR_MODE_SLAM);
        lidar_close_device(odinDevice);
        lidar_destory_device(odinDevice);
    }
    
    if (writer.joinable()) writer.join();
    
    lidar_system_deinit();
    if (pipe_fd > 0) close(pipe_fd);
    unlink(PIPE_NAME);
    
    std::cout << "Exited successfully." << std::endl;
    return 0;
}
