import os
import struct
import time
import numpy as np
import cv2
import viser

PIPE_NAME = "/tmp/odin_rgb_pipe"
MAGIC_IMAGE = 0x11223344
MAGIC_POINTCLOUD = 0x55667788

def main():
    print("Starting Viser server...")
    server = viser.ViserServer(port=8080)
    print("Viser server running on http://localhost:8080")
    
    # Configure initial camera view to match sensor forward axis
    server.initial_camera.up = (0, 0, 1)        # Z is up
    server.initial_camera.position = (-2, 0, 0.5)  # Behind origin
    server.initial_camera.look_at = (1, 0, 0)      # Looking forward along X

    
    # Create GUI image for the camera stream
    gui_image = server.gui.add_image(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        format="jpeg",
        jpeg_quality=80,
    )
    
    # Global map accumulation variables
    global_map_points = np.empty((0, 3), dtype=np.float32)
    global_map_colors = np.empty((0, 3), dtype=np.uint8)
    frame_count = 0
    
    # Simple voxel downsampling to keep browser from crashing
    def voxel_downsample(pts, cols, voxel_size=0.03):
        if len(pts) == 0:
            return pts, cols
        voxels = np.round(pts / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(voxels, axis=0, return_index=True)
        return pts[unique_indices], cols[unique_indices]

    while not os.path.exists(PIPE_NAME):
        print(f"Waiting for pipe {PIPE_NAME} to be created by C++ publisher...")
        time.sleep(1)
        
    print(f"Opening pipe {PIPE_NAME} for reading...")
    
    with open(PIPE_NAME, "rb") as pipe:
        print("Pipe opened. Waiting for frames...")
        
        while True:
            # Read 16-byte header (4 x uint32_t)
            header_bytes = pipe.read(16)
            if not header_bytes:
                print("Pipe closed by publisher. Exiting...")
                break
                
            if len(header_bytes) < 16:
                continue

            magic, arg1, arg2, length = struct.unpack("IIII", header_bytes)
            
            # Read payload
            payload = bytearray()
            while len(payload) < length:
                chunk = pipe.read(length - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)

            if len(payload) < length:
                print("Incomplete frame received. Exiting...")
                break
                
            if magic == MAGIC_IMAGE:
                # arg1=width, arg2=height
                try:
                    img_array = np.frombuffer(payload, dtype=np.uint8)
                    bgr_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if bgr_image is not None:
                        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                        gui_image.image = rgb_image
                except Exception as e:
                    print(f"Error processing image: {e}")
                    
            elif magic == MAGIC_POINTCLOUD:
                # arg1 = num_points
                try:
                    # SLAM point clouds contain 7 int32s per point: (X, Y, Z, R, G, B, A)
                    data_int32 = np.frombuffer(payload, dtype=np.int32).reshape((-1, 7))
                    
                    # X, Y, Z are in 1/10000 meters
                    points = data_int32[:, :3].astype(np.float32) / 10000.0
                    
                    # R, G, B colors
                    colors = data_int32[:, 3:6].astype(np.uint8)
                    
                    # Accumulate points
                    global_map_points = np.vstack((global_map_points, points))
                    global_map_colors = np.vstack((global_map_colors, colors))
                    frame_count += 1
                    
                    # Every 5 frames, downsample the accumulated map and push to Viser
                    if frame_count % 5 == 0:
                        global_map_points, global_map_colors = voxel_downsample(global_map_points, global_map_colors, 0.02)
                        
                        server.scene.add_point_cloud(
                            "/odin_map",
                            points=global_map_points,
                            colors=global_map_colors,
                            point_size=0.02,
                        )
                except Exception as e:
                    print(f"Error processing point cloud: {e}")
            else:
                print(f"Unknown magic number received: {hex(magic)}")

if __name__ == "__main__":
    main()
