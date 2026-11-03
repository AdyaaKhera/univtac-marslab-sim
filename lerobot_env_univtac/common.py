"""Constants shared by the converter, the sim server and the LeRobot env."""

FPS = 60             # UniVTAC save/eval frequency (task_config clean51.yml)
STATE_DIM = 8        # 7 arm joints + 1 gripper finger
IMAGE_SIZE = (128, 128)  # (H, W) for every camera

# LeRobot camera name -> HDF5 key. The sim server returns the same streams.
CAMERAS = {
    "head": "observation/head/rgb",
    "wrist": "observation/wrist/rgb",
    "tactile_left": "tactile/left_tactile/rgb_marker",
    "tactile_right": "tactile/right_tactile/rgb_marker",
}
