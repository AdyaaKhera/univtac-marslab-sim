# Progress

## Status (2026-11-03)

| Piece | State |
|---|---|
| lift_can demos → LeRobot dataset (100 eps, 28.6k frames) | done |
| LeRobot env plugin (`--env.type=univtac`) | done, registered in LeRobot |
| Diffusion Policy training (100k steps, A100) | running, loss 0.001 at 53k |
| UniVTAC env (Isaac Sim 5.1, TacEx/libuipc, cuRobo) | built, all versions check out |
| Simulation / evaluation | **blocked: needs a GPU with RT cores** |

## Blocker: Isaac Sim 5.1 rendering on Gilbreth

Isaac Sim's RTX renderer crashes at startup (`UsdManager::createHydraEngine`) on the A30 and A100 nodes available to our account. NVIDIA's own `nvcr.io/nvidia/isaac-sim:5.1.0` image crashes the same way on an A30, so it is the GPU, not our setup. NVIDIA lists RT cores as a requirement, and A30/A100/H100 do not have them.

Needs one of: Gilbreth A10 access, a cluster with L40S/RTX GPUs, or an RTX 30/40/50 workstation. Training and the trained checkpoint are unaffected.

## Log

**2026-11-02/03**
- Downloaded UniVTAC `isaac51` and the Isaac 5.1 `lift_can` demos (100 episodes, `byml/UniVTAC` on HF).
- Converted the demos to LeRobot format: head + wrist + 2 tactile (marker) cameras @128², state = joint[t][:8], action = joint[t+1][:8].
- LeRobot 0.6.1 needs py3.12 and UniVTAC needs py3.11, so the sim runs as a separate process over ZMQ.
- Gilbreth glibc 2.34 < 2.35, which the Isaac 5.1 wheels need, so the UniVTAC env runs in an Ubuntu 22.04 Apptainer image (`container/univtac.def`).
- UniVTAC install fixes (`patches/`): an nvcc version check that read the wrong line, a torch `+cu126` version check, and a GitHub-regenerated tinygltf archive (contents verified identical to tag v2.9.6).
- Started Diffusion Policy training on the A100: batch 64, crop 116², ~0.14 s/step.
