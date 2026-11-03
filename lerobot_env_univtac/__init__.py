"""LeRobot env plugin for UniVTAC. Installing this package makes
`--env.type=univtac` available to lerobot-train / lerobot-eval."""

from dataclasses import dataclass, field

import gymnasium as gym
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

from .common import CAMERAS, FPS, IMAGE_SIZE, STATE_DIM


@EnvConfig.register_subclass("univtac")
@dataclass
class UniVTACEnvConfig(EnvConfig):
    task: str = "lift_can"
    task_config: str = "clean51"
    fps: int = FPS
    features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,)),
        "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        **{f"pixels/{c}": PolicyFeature(type=FeatureType.VISUAL, shape=(*IMAGE_SIZE, 3)) for c in CAMERAS},
    })
    features_map: dict[str, str] = field(default_factory=lambda: {
        ACTION: ACTION,
        "agent_pos": OBS_STATE,
        **{f"pixels/{c}": f"{OBS_IMAGES}.{c}" for c in CAMERAS},
    })

    @property
    def gym_kwargs(self) -> dict:
        return {}

    def create_envs(self, n_envs: int, use_async_envs: bool = False):
        # Each env is a separate Isaac Sim process; UniVTAC supports one robot per process.
        from .env import UniVTACEnv

        fns = [lambda: UniVTACEnv(self.task, self.task_config) for _ in range(n_envs)]
        return {self.task: {0: gym.vector.SyncVectorEnv(fns)}}
