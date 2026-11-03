# univtac-marslab-sim

[UniVTAC](https://github.com/univtac/UniVTAC) (visuo-tactile manipulation, Isaac Sim 5.1, `isaac51` branch) run as a **LeRobot environment**, with LeRobot **Diffusion Policy** trained on UniVTAC demos.

## Results

| Task | Policy | Inputs | Demos | Success (eval seeds) |
|---|---|---|---|---|
| lift_can | Diffusion Policy | head + wrist + 2× tactile + joints | 100 | training; eval blocked (needs RTX GPU, see PROGRESS) |

Detailed log: [PROGRESS.md](PROGRESS.md)

## How it works

```
LeRobot env (py3.12, lerobot 0.6.1)          UniVTAC env (py3.11, Isaac Sim 5.1)
  lerobot-train / lerobot-eval                 sim_server.py
  --env.type=univtac  ── UniVTACEnv ──ZMQ──►   UniVTAC task (Franka + 2 GelSight Mini)
```

- `lerobot_env_univtac/`: LeRobot env plugin (`pip install -e .` registers `--env.type=univtac`)
  - `convert.py`: UniVTAC HDF5 demos → LeRobot dataset
  - `env.py`: gym env, starts and talks to the simulator
  - `sim_server.py`: runs inside the UniVTAC env
- `slurm/`: Gilbreth jobs (`env.sh` = shared paths)

Observation: `head`, `wrist`, `tactile_left`, `tactile_right` (128×128 RGB; tactile = marker images) + 8-D joint state.
Action: next 8-D joint target (7 arm + gripper), `take_action(action_type='qpos')`.

## Reproduce

```bash
# from repo root, on Gilbreth
sbatch -A <account> slurm/install_univtac.sbatch            # UniVTAC conda env
lerobot_py -m lerobot_env_univtac.convert --raw $UNIVTAC_ROOT/data/isaac51/lift_can \
  --repo-id univtac/lift_can --root $WORK/datasets/lift_can
sbatch -A <account> slurm/train_dp.sbatch lift_can
sbatch -A <account> slurm/eval_dp.sbatch $WORK/outputs/<run> lift_can
```
