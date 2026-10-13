#!/bin/bash
#SBATCH --job-name=css_genfast
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=hypatia/%j_genfast.out
#SBATCH --error=hypatia/%j_genfast.err

# Generate a CRES dataset split with BatchGenFast.py on one GPU.
#
# Step 1 is a CUDA equivalence gate: a small benchmark that compares the
# batched CUDA output with the original SignalGenerator path; if any event
# differs by more than 1e-10 (relative L2) the job stops before writing data.
# Step 2 generates the files. Arguments after the script name go straight to
# BatchGenFast.py, for example the random_phase_v1-matched test split:
#
#   sbatch hypatia/generate_fast.sh --files 10 --events 10000 --prefix test_100K \
#       --phase-seed 1020 --particle-seed 2020 --outdir $HOME/data/random_phase_v2/test
#
# Submit from a checkout of branch perf/batched-generation (Hypatia: the
# worktree ~/Github/CSS_perf). Each 10K-event 40 us file is about 3.2 GB;
# check free space on /home before large runs (it is shared).
# SKIP_GATE=1 skips step 1 once it has passed on this commit.

set -eo pipefail

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda init script not found." >&2
    exit 1
fi
conda activate CRES_Detection

cd "${SLURM_SUBMIT_DIR}"
export PYTHONPATH=$PWD
export OMP_NUM_THREADS=1        # one thread per physics worker process
WORKERS=${SLURM_CPUS_PER_TASK:-8}

echo "Job ${SLURM_JOB_ID} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES}, commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
df -h "$HOME" | tail -1

if [ "${SKIP_GATE:-0}" != "1" ]; then
    python scripts/benchmark_batched_generation.py --device cuda \
        --acq-times 40e-6 --batch-sizes 64 256 --n-events 512 --n-ref-events 8 \
        --n-workers "${WORKERS}" --threads 4 --phasors direct block \
        --filter-backends torch scipy --json "hypatia/genfast_gate_${SLURM_JOB_ID}.json"
fi

python BatchGenFast.py --device cuda --workers "${WORKERS}" --batch-size 256 "$@"
df -h "$HOME" | tail -1
