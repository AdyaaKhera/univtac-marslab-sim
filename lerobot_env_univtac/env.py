"""Gymnasium env (LeRobot side) that drives a UniVTAC sim_server subprocess."""

import json
import os
import socket
import subprocess
from pathlib import Path

import gymnasium as gym
import numpy as np
import zmq
from gymnasium import spaces

from .common import CAMERAS, IMAGE_SIZE, STATE_DIM

SERVER = Path(__file__).with_name("sim_server.py")
SEED_OFFSET = 1_000_000  # eval seeds never overlap the demo seeds 0..99, as in UniVTAC's eval


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class UniVTACEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, task: str, task_config: str = "clean51", config_overrides: tuple[str, ...] = ()):
        super().__init__()
        self.task = task  # read by lerobot's add_envs_task
        with open(Path(os.environ["UNIVTAC_ROOT"]) / "instructions" / f"{task}.json") as f:
            self.task_description = json.load(f)["seen"][0]

        port = _free_port()
        cmd = [os.environ["UNIVTAC_PYTHON"], str(SERVER), task, task_config, "--port", str(port), "--headless",
               "--image-size", *map(str, IMAGE_SIZE)]
        if config_overrides:
            cmd += ["--config-overrides", *config_overrides]
        self._proc = subprocess.Popen(cmd, cwd=os.environ["UNIVTAC_ROOT"])
        self._sock = zmq.Context.instance().socket(zmq.REQ)
        self._sock.connect(f"tcp://127.0.0.1:{port}")
        self._max_episode_steps = self._call("hello")[0]["step_lim"]  # read by lerobot_eval

        self.observation_space = spaces.Dict({
            "pixels": spaces.Dict({
                name: spaces.Box(0, 255, (*IMAGE_SIZE, 3), np.uint8) for name in CAMERAS
            }),
            "agent_pos": spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
        })
        self.action_space = spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)
        self._render = None

    def _call(self, cmd: str, **kw) -> tuple[dict, dict[str, np.ndarray]]:
        self._sock.send(json.dumps({"cmd": cmd, **kw}).encode())
        header, *blobs = self._sock.recv_multipart()
        header = json.loads(header)
        arrays = {
            k: np.frombuffer(b, dtype=dt).reshape(shape)
            for (k, dt, shape), b in zip(header.pop("arrays", []), blobs, strict=True)
        }
        return header, arrays

    def _obs(self, arrays: dict) -> dict:
        self._render = arrays["render"]
        return {
            "pixels": {name: arrays[f"pixels/{name}"] for name in CAMERAS},
            "agent_pos": arrays["agent_pos"],
        }

    def reset(self, seed: int | None = None, options=None):
        super().reset(seed=seed)
        _, arrays = self._call("reset", seed=SEED_OFFSET + int(seed or 0))
        return self._obs(arrays), {"is_success": False, "task": self.task}

    def step(self, action: np.ndarray):
        header, arrays = self._call("step", action=np.asarray(action, np.float32).reshape(-1).tolist())
        success, done = header["success"], header["done"]
        info = {"is_success": success, "task": self.task}
        if done:
            info["final_info"] = dict(info)
        return self._obs(arrays), float(success), success, done and not success, info

    def render(self) -> np.ndarray:
        return self._render

    def close(self):
        if self._proc.poll() is None:
            try:
                self._call("close")
                self._proc.wait(timeout=120)
            except Exception:
                self._proc.kill()
