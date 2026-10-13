#!/bin/bash
#SBATCH --job-name=css_batched_bench
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Benchmark and equivalence check for BatchedSignalGenerator on one V100.
#
# Runs scripts/benchmark_batched_generation.py with --device cuda: times the
# original SignalGenerator path (commit 3cf55ca, extracted with git archive),
# the cached CPU path and the batched path (physics on 4 CPU workers,
# synthesis on the GPU) at 40 us and 200 us for several batch sizes, and
# checks every batched configuration against SignalGenerator (relative L2
# per event <= 1e-10; the script exits non-zero otherwise). The batched unit
# tests are run first on the CPU.
#
# Requirements:
#   * a checkout of the CRESSignalStructure repository on Hypatia at the
#     perf/batched-generation branch (or later), including commit 3cf55ca;
#     submit from its root, since the job cds to $SLURM_SUBMIT_DIR and sets
#     PYTHONPATH to it
#   * PyTorch (>= 2.0, CUDA build), numpy, scipy, h5py and pytest in the
#     conda env CRES_Detection
#
# Usage (from the repository root on the Hypatia login node):
#   sbatch hypatia/benchmark_batched_generation.sh
#   sbatch hypatia/benchmark_batched_generation.sh --batch-sizes 16 64   # extra args pass through

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
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}

echo "=========================================="
echo "CRESSignalStructure batched generation benchmark"
echo "Job ID:   ${SLURM_JOB_ID}"
echo "Node:     $(hostname)"
echo "GPU:      ${CUDA_VISIBLE_DEVICES}"
echo "Commit:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Branch:   $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import CRESSignalStructure, torch; print(CRESSignalStructure.__file__); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python -m pytest -q -p no:cacheprovider tests/unit/test_batched_signal_generator.py

OUT_JSON="benchmark_batched_${SLURM_JOB_ID}.json"
python scripts/benchmark_batched_generation.py \
    --device cuda \
    --acq-times 40e-6 200e-6 \
    --batch-sizes 8 32 64 \
    --n-events 256 \
    --n-ref-events 4 \
    --n-workers "${OMP_NUM_THREADS}" \
    --threads "${OMP_NUM_THREADS}" \
    --phasors direct block \
    --filter-backends torch scipy \
    --json "${OUT_JSON}" \
    "$@"

echo "Results written to ${OUT_JSON}"
