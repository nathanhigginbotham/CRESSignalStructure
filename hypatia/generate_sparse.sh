#!/bin/bash
#SBATCH --job-name=css_gensparse
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=12GB
#SBATCH --time=24:00:00
#SBATCH --output=hypatia/%j_gensparse.out
#SBATCH --error=hypatia/%j_gensparse.err

# Generate sparse noiseless spectra with BatchGenFast.py --sparse on one GPU.
#
# Usage: sbatch hypatia/generate_sparse.sh OUTDIR [BatchGenFast.py arguments]
#
# OUTDIR may be on /home (e.g. $HOME/data/sparse_1M/train) or node-local
# (/state/partition1/$USER/...; compute-node /tmp is too small). Node-local
# output must be copied off before the job ends; that is left to the caller.
#
# Each file index becomes one directory <prefix>_<fff>_sparse (written as
# .partial and renamed), about 45-50 kB per 40 us event, so a 10K-event
# file is ~0.5 GB and 1M events ~47 GB. The layout is the detection repo's
# full-band cache; scripts/merge_sparse_chunks.py joins the per-file
# directories into the single directory its selector trainer reads.
#
# Step 1 is the CUDA equivalence gate of generate_fast.sh (stops the job if
# the batched CUDA records differ from SignalGenerator by more than 1e-10
# relative L2); SKIP_GATE=1 skips it once it has passed on this commit.
# Step 2 is a small sparse-vs-dense validation (SKIP_CHECK=1 skips it).
# Both steps run at the geometry of the run: any --trap-depth, --pitch-min,
# --pitch-max, --lo-mode or --lo-freq among the BatchGenFast.py arguments is
# passed to them too, so a pass at 4 mT does not stand in for a 32 mT run.
#
# Broad range (32 mT trap, pitch 80-90 deg, carrier centred in the band):
#   sbatch hypatia/generate_sparse.sh $HOME/data/broad/train --trap-depth 32e-3 \
#       --pitch-min 80 --pitch-max 90 --lo-mode centre ...
#
# Splitting 1M events across jobs: every event is seeded by
# [particle_seed, file_index, i] (phases likewise), so file f is the same
# whichever job writes it, and jobs are independent. Give each job its own
# --first-file range with the same seeds and prefix, e.g. 100 files of 10K
# events as 4 jobs of 25 files:
#
#   for k in 0 1 2 3; do
#     sbatch hypatia/generate_sparse.sh $HOME/data/sparse_1M/train \
#         --first-file $((25 * k)) --files 25 --events 10000 --prefix train \
#         --phase-seed 1040 --particle-seed 2040   # never reuse the val (1010/2010) or test (1020/2020) seeds
#   done
#
# Calibrate --time from the ev/s that the first job logs per file; a job
# that hits its time limit is resubmitted unchanged (finished files are
# skipped, the interrupted .partial is redone). Keep the Hypatia courtesy
# limit of about 4 concurrent jobs in mind.

set -eo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch hypatia/generate_sparse.sh OUTDIR [BatchGenFast.py arguments]" >&2
    exit 1
fi
OUTDIR="$1"
shift

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

mkdir -p "$OUTDIR"
echo "Job ${SLURM_JOB_ID} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES}, commit $(git rev-parse --short HEAD)"
echo "Output: $OUTDIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
df -h "$OUTDIR" | tail -1

# The physics options of the run, forwarded to the gate and the check
PHYS_ARGS=()
ARGV=("$@")
for ((i = 0; i < ${#ARGV[@]}; i++)); do
    case "${ARGV[i]}" in
        --trap-depth|--pitch-min|--pitch-max|--lo-mode|--lo-freq)
            PHYS_ARGS+=("${ARGV[i]}" "${ARGV[i + 1]}"); i=$((i + 1)) ;;
        --trap-depth=*|--pitch-min=*|--pitch-max=*|--lo-mode=*|--lo-freq=*)
            PHYS_ARGS+=("${ARGV[i]}") ;;
    esac
done
echo "Physics options for the gate and check: ${PHYS_ARGS[*]:-(defaults)}"

if [ "${SKIP_GATE:-0}" != "1" ]; then
    python scripts/benchmark_batched_generation.py --device cuda \
        --acq-times 40e-6 --batch-sizes 64 256 --n-events 512 --n-ref-events 8 \
        --n-workers "${WORKERS}" --threads 4 --phasors direct block \
        --filter-backends torch scipy --json "hypatia/gensparse_gate_${SLURM_JOB_ID}.json" \
        "${PHYS_ARGS[@]}"
fi

if [ "${SKIP_CHECK:-0}" != "1" ]; then
    # Sparse against dense on CUDA, in a scratch directory that is deleted after
    python scripts/check_sparse_generation.py --n-events 256 --device cuda \
        --workers "${WORKERS}" --batch-size 128 \
        --json "hypatia/gensparse_check_${SLURM_JOB_ID}.json" -- "${PHYS_ARGS[@]}"
fi

python BatchGenFast.py --device cuda --workers "${WORKERS}" --batch-size 256 \
    --sparse --outdir "$OUTDIR" "$@"
df -h "$OUTDIR" | tail -1
