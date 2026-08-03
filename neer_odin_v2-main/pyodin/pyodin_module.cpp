// pyodin_module.cpp
//
// Minimal pybind11 wrapper around Manifold's Odin 1 host SDK (lidar_api.h),
// which ships as a static archive with a C ABI (liblydHostApi_{arm,amd}.a).
// This lets a pure-Python publisher (robot/odin_pub_node.py) and the standalone
// Viser test (tools/odin_viser_test.py) drive the box WITHOUT ROS.
//
// Design:
//   * SDK data callbacks run on internal SDK threads and only touch C++ buffers
//     (no Python / no GIL) -> they just copy the latest frame per type under a
//     mutex. Pointers from the SDK are only valid during the callback, so we
//     copy immediately (see lidar_api.h "Data Lifetime" note).
//   * Python-facing getters lock the mutex, copy out, unlock, then build numpy
//     arrays (GIL held by the caller). Returns None until the first frame.
//   * lidar_system_init()'s device callback has NO user_data arg, so the
//     instance is reached through a single global pointer (g_self). One sensor
//     per process (matches the hardware: one Odin box).
//
// Sequence mirrors the proven prototype (~/odin_pipe_publisher.cpp) and
// odin_ros_driver/include/host_sdk_sample.h.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "lidar_api.h"
#include "lidar_api_type.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

// ---- forward declarations (callbacks must be passable to the C SDK) ----
class OdinSensor;
static OdinSensor* g_self = nullptr;
static void device_cb_free(const lidar_device_info_t* info, bool attach);
static void data_cb_free(const lidar_data_t* data, void* user);

// ---- latest-frame buffers ----
struct RgbFrame {
    uint64_t ts = 0;
    uint32_t w = 0, h = 0;
    bool is_jpeg = false;             // false => NV12 (length == w*h*3/2)
    std::vector<uint8_t> data;
    bool valid = false;
};
struct CloudFrame {
    uint64_t ts = 0;
    uint32_t n = 0;
    std::vector<int32_t> data;        // n * 7 int32 (xyz + rgba), see slam_cloud_point_t
    bool valid = false;
};
struct DepthFrame {
    uint64_t ts = 0;
    uint32_t w = 0, h = 0;
    std::vector<float> data;          // meters, 256x192
    bool valid = false;
};
struct DtofCloudFrame {               // organized dToF cloud (camera frame, 256x192)
    uint64_t ts = 0;
    uint32_t w = 0, h = 0;
    std::vector<float> xyz;           // h*w*3 interleaved (meters)
    std::vector<uint8_t> conf;        // h*w confidence (0..255)
    bool valid = false;
};
struct OdomFrame {
    uint64_t ts = 0;
    double pos[3]  = {0, 0, 0};       // meters
    double quat[4] = {0, 0, 0, 1};    // x, y, z, w
    double lin[3]  = {0, 0, 0};
    double ang[3]  = {0, 0, 0};
    bool valid = false;
};
struct ImuFrame {
    uint64_t ts = 0;
    float accel[3] = {0, 0, 0};       // m/s^2
    float gyro[3]  = {0, 0, 0};       // rad/s
    bool valid = false;
};

class OdinSensor {
public:
    OdinSensor() = default;
    ~OdinSensor() { stop(); }

    // Initialise the SDK and block (GIL released) until the device attaches and
    // streams start, or wait_s elapses. mode: "slam" (default) or "raw".
    bool start(const std::string& mode, double wait_s, int map_mode) {
        if (running_) return connected_;
        slam_mode_ = (mode != "raw");
        map_mode_ = map_mode;
        keep_running_ = true;
        connected_ = false;
        g_self = this;
        if (lidar_system_init(&device_cb_free) != 0) {
            g_self = nullptr;
            return false;
        }
        running_ = true;
        {
            py::gil_scoped_release rel;  // the attach callback runs on an SDK thread
            auto t0 = std::chrono::steady_clock::now();
            while (keep_running_ && !connected_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                double el = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0).count();
                if (el > wait_s) break;
            }
        }
        return connected_;
    }

    void stop() {
        keep_running_ = false;
        if (running_) {
            if (connected_ && dev_) {
                lidar_stop_stream(dev_, slam_mode_ ? LIDAR_MODE_SLAM : LIDAR_MODE_RAW);
                lidar_close_device(dev_);
                lidar_destory_device(dev_);
            }
            lidar_system_deinit();
            running_ = false;
            connected_ = false;
            dev_ = nullptr;
        }
        if (g_self == this) g_self = nullptr;
    }

    bool connected() const { return connected_; }

    py::dict counts() const {
        py::dict d;
        d["rgb"]   = rgb_count_.load();
        d["cloud"] = cloud_count_.load();
        d["odom"]  = odom_count_.load();
        d["depth"] = depth_count_.load();
        d["imu"]   = imu_count_.load();
        return d;
    }

    // ---------- called from SDK threads (C++ only, no Python) ----------
    void on_device(const lidar_device_info_t* info, bool attach) {
        if (!attach) { connected_ = false; return; }

        if (lidar_create_device((lidar_device_info_t*)info, &dev_) != 0) return;

        lidar_data_callback_info_t cb;
        cb.data_callback = &data_cb_free;
        cb.user_data     = this;
        if (lidar_register_stream_callback(dev_, cb) != 0) return;
        if (lidar_open_device(dev_) != 0) return;

        int mode = slam_mode_ ? LIDAR_MODE_SLAM : LIDAR_MODE_RAW;
        lidar_set_mode(dev_, mode);

        if (slam_mode_) {
            // custom_map_mode: 0=odometry (no loop closure, drifts),
            // 1=mapping (loop closure + saveable map), 2=relocalization.
            // Default 1 so the cloud is properly registered into a clean map.
            int mm = map_mode_;
            lidar_set_custom_parameter(dev_, "custom_map_mode", &mm, sizeof(int));
        }

        lidar_activate_stream_type(dev_, LIDAR_DT_RAW_RGB);
        lidar_activate_stream_type(dev_, LIDAR_DT_RAW_IMU);
        lidar_activate_stream_type(dev_, LIDAR_DT_RAW_DTOF);
        if (slam_mode_) {
            lidar_activate_stream_type(dev_, LIDAR_DT_SLAM_CLOUD);
            lidar_activate_stream_type(dev_, LIDAR_DT_SLAM_ODOMETRY);
        }

        uint32_t odr = 0;
        // Start the SLAM-mode stream set: rgb, SLAM cloud, odometry, imu.
        // NOTE (verified on hardware): raw DTOF does NOT stream concurrently with the
        // SLAM products on this firmware — starting LIDAR_DT_RAW_DTOF explicitly stalls
        // the SLAM cloud + odometry. So zed/pcd is sourced from the SLAM cloud
        // (odin_pub_node.py), and get_dtof_cloud() stays empty in SLAM mode.
        lidar_start_stream(dev_, mode, odr);
        connected_ = true;
        // Calibration is intentionally NOT fetched here: at attach the device is not yet
        // streaming and lidar_get_calibration returns zeros. get_calibration() fetches it
        // live (and only caches once the device returns real, non-zero intrinsics).
    }

    void on_data(const lidar_data_t* data) {
        if (!data || data->stream.imageCount < 1) return;
        const buffer_List_t& im = data->stream.imageList[0];
        if (!im.pAddr || im.length == 0) return;

        switch (data->type) {
        case LIDAR_DT_RAW_RGB: {
            std::lock_guard<std::mutex> lk(mtx_);
            rgb_.ts = im.timestamp;
            rgb_.w = im.width;
            rgb_.h = im.height;
            // NV12 iff length matches the planar YUV size, else JPEG (host_sdk_sample.h:806)
            rgb_.is_jpeg = (im.length != (uint64_t)im.width * im.height * 3 / 2);
            rgb_.data.assign((const uint8_t*)im.pAddr, (const uint8_t*)im.pAddr + im.length);
            rgb_.valid = true;
            rgb_count_++;
            break;
        }
        case LIDAR_DT_SLAM_CLOUD: {
            uint32_t n = im.length / (7 * sizeof(int32_t));  // slam_cloud_point_t = 28 bytes
            std::lock_guard<std::mutex> lk(mtx_);
            cloud_.ts = im.timestamp;
            cloud_.n = n;
            cloud_.data.resize((size_t)n * 7);
            std::memcpy(cloud_.data.data(), im.pAddr, (size_t)n * 7 * sizeof(int32_t));
            cloud_.valid = true;
            cloud_count_++;
            break;
        }
        case LIDAR_DT_SLAM_ODOMETRY: {
            if (im.length < sizeof(ros_odom_convert_complete_t)) break;
            const auto* o = (const ros_odom_convert_complete_t*)im.pAddr;
            std::lock_guard<std::mutex> lk(mtx_);
            odom_.ts = o->timestamp_ns;
            for (int i = 0; i < 3; ++i) odom_.pos[i] = o->pos[i] / 1.0e6;          // um -> m
            for (int i = 0; i < 4; ++i) odom_.quat[i] = o->orient[i] / 1.0e6;       // x,y,z,w
            for (int i = 0; i < 3; ++i) odom_.lin[i] = o->linear_velocity[i] / 1.0e6;
            for (int i = 0; i < 3; ++i) odom_.ang[i] = o->angular_velocity[i] / 1.0e6;
            odom_.valid = true;
            odom_count_++;
            break;
        }
        case LIDAR_DT_RAW_DTOF: {
            // imageCount==4: [0]=depth(float m), [1]=organized xyz(float), [2]=conf(u8), [3]=intensity(u16)
            const auto& s = data->stream;
            std::lock_guard<std::mutex> lk(mtx_);
            // depth image (256x192 float meters)
            const buffer_List_t& d = s.imageList[0];
            if (d.pAddr && d.length) {
                uint32_t n = d.length / sizeof(float);
                depth_.ts = d.timestamp; depth_.w = d.width; depth_.h = d.height;
                depth_.data.resize(n);
                std::memcpy(depth_.data.data(), d.pAddr, (size_t)n * sizeof(float));
                depth_.valid = true;
            }
            // organized XYZ point cloud (camera frame) — the ZED-pcd analog
            if (s.imageCount >= 2) {
                const buffer_List_t& p = s.imageList[1];
                if (p.pAddr && p.length) {
                    uint32_t n = p.length / sizeof(float);
                    dtof_.ts = p.timestamp; dtof_.w = p.width; dtof_.h = p.height;
                    dtof_.xyz.resize(n);
                    std::memcpy(dtof_.xyz.data(), p.pAddr, (size_t)n * sizeof(float));
                    dtof_.conf.clear();
                    if (s.imageCount >= 3) {
                        const buffer_List_t& cf = s.imageList[2];
                        if (cf.pAddr && cf.length) {
                            dtof_.conf.resize(cf.length);
                            std::memcpy(dtof_.conf.data(), cf.pAddr, cf.length);
                        }
                    }
                    dtof_.valid = true;
                }
            }
            depth_count_++;
            break;
        }
        case LIDAR_DT_RAW_IMU: {
            if (im.length < sizeof(imu_convert_data_t)) break;
            const auto* u = (const imu_convert_data_t*)im.pAddr;
            std::lock_guard<std::mutex> lk(mtx_);
            imu_.ts = u->stamp;
            imu_.accel[0] = u->accel_x; imu_.accel[1] = u->accel_y; imu_.accel[2] = u->accel_z;
            imu_.gyro[0]  = u->gyro_x;  imu_.gyro[1]  = u->gyro_y;  imu_.gyro[2]  = u->gyro_z;
            imu_.valid = true;
            imu_count_++;
            break;
        }
        default:
            break;
        }
    }

    // ---------- Python-facing getters (GIL held by caller) ----------
    py::object get_rgb() {
        RgbFrame f;
        { std::lock_guard<std::mutex> lk(mtx_); if (!rgb_.valid) return py::none(); f = rgb_; }
        py::dict d;
        d["timestamp"] = f.ts;
        d["width"] = f.w;
        d["height"] = f.h;
        d["is_jpeg"] = f.is_jpeg;
        d["data"] = py::array_t<uint8_t>((py::ssize_t)f.data.size(), f.data.data());
        return d;
    }

    py::object get_cloud() {
        CloudFrame f;
        { std::lock_guard<std::mutex> lk(mtx_); if (!cloud_.valid) return py::none(); f = cloud_; }
        auto arr = py::array_t<int32_t>({(py::ssize_t)f.n, (py::ssize_t)7});
        std::memcpy(arr.mutable_data(), f.data.data(), f.data.size() * sizeof(int32_t));
        py::dict d;
        d["timestamp"] = f.ts;
        d["n"] = f.n;
        d["xyzrgba"] = arr;   // (n,7) int32: xyz in 0.1mm, rgba in low bytes
        return d;
    }

    py::object get_odom() {
        OdomFrame f;
        { std::lock_guard<std::mutex> lk(mtx_); if (!odom_.valid) return py::none(); f = odom_; }
        py::dict d;
        d["timestamp_ns"] = f.ts;
        d["pos"]      = py::array_t<double>(3, f.pos);    // meters (x,y,z)
        d["quat_xyzw"] = py::array_t<double>(4, f.quat);
        d["lin_vel"]  = py::array_t<double>(3, f.lin);
        d["ang_vel"]  = py::array_t<double>(3, f.ang);
        return d;
    }

    py::object get_depth() {
        DepthFrame f;
        { std::lock_guard<std::mutex> lk(mtx_); if (!depth_.valid) return py::none(); f = depth_; }
        auto arr = py::array_t<float>({(py::ssize_t)f.h, (py::ssize_t)f.w});
        std::memcpy(arr.mutable_data(), f.data.data(), f.data.size() * sizeof(float));
        py::dict d;
        d["timestamp"] = f.ts;
        d["width"] = f.w;
        d["height"] = f.h;
        d["depth"] = arr;
        return d;
    }

    py::object get_dtof_cloud() {
        DtofCloudFrame f;
        { std::lock_guard<std::mutex> lk(mtx_); if (!dtof_.valid) return py::none(); f = dtof_; }
        auto xyz = py::array_t<float>({(py::ssize_t)f.h, (py::ssize_t)f.w, (py::ssize_t)3});
        std::memcpy(xyz.mutable_data(), f.xyz.data(), f.xyz.size() * sizeof(float));
        py::dict d;
        d["timestamp"] = f.ts;
        d["width"] = f.w;
        d["height"] = f.h;
        d["xyz"] = xyz;   // (h,w,3) float meters, camera frame
        if (!f.conf.empty()) {
            auto cf = py::array_t<uint8_t>({(py::ssize_t)f.h, (py::ssize_t)f.w});
            std::memcpy(cf.mutable_data(), f.conf.data(), f.conf.size());
            d["confidence"] = cf;
        } else {
            d["confidence"] = py::none();
        }
        return d;
    }

    py::object get_imu() {
        ImuFrame f;
        { std::lock_guard<std::mutex> lk(mtx_); if (!imu_.valid) return py::none(); f = imu_; }
        py::dict d;
        d["timestamp"] = f.ts;
        d["accel"] = py::array_t<float>(3, f.accel);
        d["gyro"]  = py::array_t<float>(3, f.gyro);
        return d;
    }

    py::object get_calibration() {
        lidar_calibration_t c;
        bool ok = false;
        { std::lock_guard<std::mutex> lk(mtx_); if (calib_valid_) { c = calib_; ok = true; } }
        if (!ok && dev_ && lidar_get_calibration(dev_, &c) == 0) {
            ok = true;
            if (c.intrinsics[0] != 0.0f) {  // only cache once the device returns real values
                std::lock_guard<std::mutex> lk(mtx_);
                calib_ = c; calib_valid_ = true;
            }
        }
        if (!ok) return py::none();
        py::dict d;
        auto K = py::array_t<float>(9);  std::memcpy(K.mutable_data(), c.intrinsics, 9 * sizeof(float));
        auto E = py::array_t<float>(16); std::memcpy(E.mutable_data(), c.extrinsics, 16 * sizeof(float));
        d["intrinsics"] = K;     // 3x3 row-major K
        d["extrinsics"] = E;     // 4x4 row-major (cam <-> imu/lidar)
        d["fx"] = c.intrinsics[0];
        d["fy"] = c.intrinsics[4];
        d["cx"] = c.intrinsics[2];
        d["cy"] = c.intrinsics[5];
        { std::lock_guard<std::mutex> lk(mtx_);
          d["width"]  = rgb_.valid ? rgb_.w : 0u;
          d["height"] = rgb_.valid ? rgb_.h : 0u; }
        return d;
    }

    bool get_calib_file(const std::string& path) {
        return dev_ && lidar_get_calib_file(dev_, path.c_str()) == 0;
    }
    int set_custom_parameter(const std::string& name, int value) {
        if (!dev_) return -1;
        return lidar_set_custom_parameter(dev_, name.c_str(), &value, sizeof(int));
    }
    int save_map(const std::string& dest_dir, const std::string& file_name, uint32_t timeout_ms) {
        if (!dev_) return -1;
        return lidar_save_map(dev_, dest_dir.c_str(), file_name.c_str(), timeout_ms);
    }
    int set_relocalization_map(const std::string& path) {
        if (!dev_) return -1;
        return lidar_set_relocalization_map(dev_, path.c_str());
    }

private:
    device_handle dev_ = nullptr;
    bool slam_mode_ = true;
    int map_mode_ = 1;
    std::atomic<bool> running_{false};
    std::atomic<bool> connected_{false};
    std::atomic<bool> keep_running_{true};

    std::mutex mtx_;
    RgbFrame rgb_;
    CloudFrame cloud_;
    DepthFrame depth_;
    DtofCloudFrame dtof_;
    OdomFrame odom_;
    ImuFrame imu_;
    lidar_calibration_t calib_{};
    bool calib_valid_ = false;

    std::atomic<uint64_t> rgb_count_{0}, cloud_count_{0}, odom_count_{0},
                          depth_count_{0}, imu_count_{0};
};

// ---- C-callable trampolines ----
static void device_cb_free(const lidar_device_info_t* info, bool attach) {
    if (g_self) g_self->on_device(info, attach);
}
static void data_cb_free(const lidar_data_t* data, void* user) {
    if (user) reinterpret_cast<OdinSensor*>(user)->on_data(data);
}

PYBIND11_MODULE(pyodin, m) {
    m.doc() = "Non-ROS Python bridge to the Manifold Odin 1 host SDK";
    py::class_<OdinSensor>(m, "OdinSensor")
        .def(py::init<>())
        .def("start", &OdinSensor::start, py::arg("mode") = "slam", py::arg("wait_s") = 10.0,
             py::arg("map_mode") = 1,
             "Init SDK, activate+start streams, block until connected or timeout. "
             "map_mode: 0=odometry, 1=mapping+loop-closure (default), 2=relocalization.")
        .def("stop", &OdinSensor::stop)
        .def("connected", &OdinSensor::connected)
        .def("counts", &OdinSensor::counts)
        .def("get_rgb", &OdinSensor::get_rgb, "Latest RGB: {timestamp,width,height,is_jpeg,data} or None")
        .def("get_cloud", &OdinSensor::get_cloud, "Latest SLAM cloud: {timestamp,n,xyzrgba(n,7 int32)} or None")
        .def("get_odom", &OdinSensor::get_odom, "Latest odometry: {timestamp_ns,pos(m),quat_xyzw,...} or None")
        .def("get_depth", &OdinSensor::get_depth, "Latest dToF depth (m) or None")
        .def("get_dtof_cloud", &OdinSensor::get_dtof_cloud,
             "Latest organized dToF cloud {timestamp,width,height,xyz(h,w,3),confidence} or None")
        .def("get_imu", &OdinSensor::get_imu, "Latest IMU or None")
        .def("get_calibration", &OdinSensor::get_calibration, "Intrinsics K + extrinsics, or None")
        .def("get_calib_file", &OdinSensor::get_calib_file, py::arg("path"),
             "Pull the device calib.yaml (fisheye polynomial) to path")
        .def("set_custom_parameter", &OdinSensor::set_custom_parameter,
             py::arg("name"), py::arg("value"),
             "Set an int device parameter (e.g. custom_map_mode).")
        .def("save_map", &OdinSensor::save_map,
             py::arg("dest_dir"), py::arg("file_name"), py::arg("timeout_ms") = 0)
        .def("set_relocalization_map", &OdinSensor::set_relocalization_map, py::arg("path"));
}
