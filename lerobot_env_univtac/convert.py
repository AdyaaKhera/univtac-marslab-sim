"""Convert UniVTAC HDF5 demos into a LeRobot dataset.

state[t] = joint[t][:8], action[t] = joint[t+1][:8]  (7 arm joints + 1 finger),
matching UniVTAC's own preprocessors and task.take_action(action_type='qpos').
Images are kept in the channel order stored in the HDF5, which is the same
order the simulator returns at eval time.

Usage (lerobot env):
    python -m lerobot_env_univtac.convert --raw $UNIVTAC_ROOT/data/isaac51/lift_can \
        --repo-id univtac/lift_can --root $WORK/datasets/lift_can
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from .common import CAMERAS, FPS, IMAGE_SIZE, STATE_DIM


def decode(buf: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(img, IMAGE_SIZE[::-1], interpolation=cv2.INTER_AREA)


def features() -> dict:
    feats = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
    }
    for name in CAMERAS:
        feats[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (*IMAGE_SIZE, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, required=True, help="folder with hdf5/*.hdf5")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=None, help="first N episodes (default all)")
    p.add_argument("--task", default="lift the can")
    args = p.parse_args()

    files = sorted((args.raw / "hdf5").glob("*.hdf5"), key=lambda f: int(f.stem))[: args.episodes]
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=FPS, features=features(), root=args.root,
        robot_type="franka_gsmini", use_videos=True, image_writer_threads=8,
    )
    for i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            joint = f["embodiment/joint"][:, :STATE_DIM].astype(np.float32)
            streams = {name: f[key] for name, key in CAMERAS.items()}
            for t in range(len(joint) - 1):
                frame = {"observation.state": joint[t], "action": joint[t + 1], "task": args.task}
                for name, data in streams.items():
                    frame[f"observation.images.{name}"] = decode(data[t])
                ds.add_frame(frame)
        ds.save_episode()
        print(f"[{i + 1}/{len(files)}] {path.name}: {len(joint) - 1} frames", flush=True)
    ds.finalize()


if __name__ == "__main__":
    main()
