#!/bin/bash
#SBATCH --job-name=css_genscatter
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --output=hypatia/%j_genscatter.out
#SBATCH --error=hypatia/%j_genscatter.err

# Generate a gas-scattering ensemble with BatchGenScattering.py on CPU.
#
# Usage: sbatch hypatia/generate_scattering.sh OUTDIR [BatchGenScattering.py arguments]
#
# With the seeds of a BatchGenFast.py run, event i is the same electron and
# phases as event i of that run's file --file-index. The electrons of
# random_phase_v2 test file 000 (never used for training):
#   sbatch hypatia/generate_scattering.sh $HOME/data/scattered_v2 \
#       --events 1500 --prefix scattered_v2 --particle-seed 2020 \
#       --phase-seed 1020 --file-index 0
#
# Step 1 (SKIP_CHECK=1 skips it) runs scripts/check_scattering_generation.py
# on 12 events at the default box: with the gas off each event must match
# SignalGenerator, and with it on the waveform before the first scatter
# must match the unscattered one. The job stops if it fails.
#
# Measured on a laptop core: 0.29 events/s at 1e12 cm^-3 and 500 us, so
# 1500 events take about 15 minutes on 8 workers. Events are up to 500 us
# (4 MB as complex64); 1500 events need about 4 GB.

set -eo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: sbatch hypatia/generate_scattering.sh OUTDIR [BatchGenScattering.py arguments]" >&2
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
export OMP_NUM_THREADS=1        # one thread per worker process
WORKERS=${SLURM_CPUS_PER_TASK:-8}

mkdir -p "$OUTDIR"
echo "Job ${SLURM_JOB_ID} on $(hostname), commit $(git rev-parse --short HEAD)"
echo "Output: $OUTDIR"
df -h "$OUTDIR" | tail -1

if [ "${SKIP_CHECK:-0}" != "1" ]; then
    python scripts/check_scattering_generation.py --particle-seed 2020 \
        --phase-seed 1020 --file-index 0 --events 12 \
        --expect-lo 26667465723.059723 2>&1 | grep -v "^20[0-9][0-9]-"
fi

python BatchGenScattering.py --outdir "$OUTDIR" --workers "$WORKERS" "$@"
ls -la "$OUTDIR"
