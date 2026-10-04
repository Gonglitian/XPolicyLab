# v1 baseline code as it runs on UCR BCC (copied 2026-10-03)
Starting point for merging the v1 baseline pipeline into XPolicyLab (Week 2 task for Hanyun).
Differences from ../code/ (the labserver copy): paths are overridable through V1_* environment variables
(see bcc_env.sh), old checkpoints can be pruned (V1_PRUNE_CHECKPOINTS=1), bcc_lanes.sbatch runs four
single-GPU lanes inside one Slurm job, launch_overlap.sh adds a lane to a running job.
Training and evaluation logic (train_stage.py, evaluate.py, serve.py, prepare_*.py) is byte-identical to ../code/.
Do not start lanes from this directory on labserver; all eight baseline streams run on BCC.
