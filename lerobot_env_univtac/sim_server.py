"""UniVTAC simulator served over ZMQ. Runs in the UniVTAC (Isaac Sim 5.1) env.

Started by lerobot_env_univtac.env.UniVTACEnv; not meant to be run by hand
except for debugging:
    cd $UNIVTAC_ROOT && python <this file> lift_can clean51 --port 5555 --headless

Protocol (REQ/REP). Request: JSON {"cmd": ..., ...}. Reply: multipart
[JSON header, raw array bytes...]; the header lists each array's key/dtype/shape.
Raw bytes avoid pickling numpy across the two envs' different numpy versions.
  hello            -> {"step_lim": int}
  reset {seed}     -> obs
  step {action}    -> obs + {"success", "done"}
  close            -> {}
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path[:0] = [".", "./policy"]  # cwd is the UniVTAC repo root

from envs.utils.env_parser import add_config_override_argument, create_task_env, load_task_config
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("task_name")
parser.add_argument("task_config")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--image-size", type=int, nargs=2, default=(128, 128), help="H W")
parser.add_argument("--render-size", type=int, nargs=2, default=(270, 480), help="H W of render frame")
parser.add_argument("--save-dir", type=Path, default=None, help="UniVTAC eval videos/logs")
add_config_override_argument(parser)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.num_envs = 1

task_config, task_config_file = load_task_config(args.task_config, args.config_overrides)
simulation_app = AppLauncher(args).app  # must start before importing omni/isaac modules

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import zmq  # noqa: E402

# Same streams as lerobot_env_univtac.common.CAMERAS: (root, sensor, key)
CAMERAS = {
    "head": ("observation", "head", "rgb"),
    "wrist": ("observation", "wrist", "rgb"),
    "tactile_left": ("tactile", "left_tactile", "rgb_marker"),
    "tactile_right": ("tactile", "right_tactile", "rgb_marker"),
}


def to_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def pack(obs: dict, **extra) -> list[bytes]:
    h, w = args.image_size
    arrays = {"agent_pos": to_numpy(obs["embodiment"]["joint"])[:8].astype(np.float32)}
    for name, (root, sensor, key) in CAMERAS.items():
        img = to_numpy(obs[root][sensor][key]).astype(np.uint8)
        arrays[f"pixels/{name}"] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        if name == "head":
            rh, rw = args.render_size
            arrays["render"] = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    header = {**extra, "arrays": [[k, str(a.dtype), list(a.shape)] for k, a in arrays.items()]}
    return [json.dumps(header).encode()] + [np.ascontiguousarray(a).tobytes() for a in arrays.values()]


def main():
    save_dir = args.save_dir or Path("eval_result/lerobot") / args.task_name / time.strftime("%Y-%m-%d_%H-%M-%S")
    task = create_task_env(
        args.task_name, task_config, task_config_file.stem, "eval", device=args.device, save_dir=save_dir
    ).task
    with open(f"instructions/{args.task_name}.json") as f:
        instructions = json.load(f)["seen"]

    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[sim_server] ready on port {args.port}", flush=True)

    in_episode = False
    while True:
        req = json.loads(sock.recv())
        cmd = req["cmd"]
        if cmd == "hello":
            sock.send_multipart([json.dumps({"step_lim": int(task.cfg.step_lim)}).encode()])
        elif cmd == "reset":
            if in_episode:
                task.clean_cache(result="failed")
            task.mode = "eval"
            task.reset(seed=int(req["seed"]), instructions=instructions)
            task.mean_steps = task.cfg.step_lim
            in_episode = True
            sock.send_multipart(pack(task._get_observations()))
        elif cmd == "step":
            action = torch.tensor(req["action"], dtype=torch.float32, device=task.device)
            task.take_action(action, action_type="qpos")
            success = bool(task.eval_success)
            done = success or task.take_action_cnt >= task.cfg.step_lim or bool(task.check_early_stop())
            if done and in_episode:
                task.clean_cache(result="success" if success else "failed")  # writes UniVTAC's video
                in_episode = False
            sock.send_multipart(pack(task._get_observations(), success=success, done=done))
        elif cmd == "close":
            sock.send_multipart([b"{}"])
            break
    task.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
