#!/bin/bash
#SBATCH --job-name=era5_test
#SBATCH --output=logs/era5_test.out   # relative to submission directory (CDHW_ag/Code/)
#SBATCH --error=logs/era5_test.err
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --export=ALL                  # inherits ERA5_dat, MAMBA_EXE, MAMBA_ROOT_PREFIX from ~/.bashrc

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ─────────────────────────────────────────────────────────────────────────────

# SLURM_SUBMIT_DIR is always set to the directory where sbatch was called from
SCRIPT_DIR=$SLURM_SUBMIT_DIR

# Initialise mamba using MAMBA_EXE and MAMBA_ROOT_PREFIX inherited from ~/.bashrc
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX)"
mamba activate CE_env

# ─────────────────────────────────────────────────────────────────────────────
# TEST SETTINGS
# ─────────────────────────────────────────────────────────────────────────────

TEST_YEAR=2020
TEST_OUTPUT_DIR="${ERA5_dat}_test"    # separate folder so test data never mixes with production

mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$TEST_OUTPUT_DIR"

echo "========================================"
echo "SLURM job:    $SLURM_JOB_ID"
echo "Node:         $SLURMD_NODENAME"
echo "Script dir:   $SCRIPT_DIR"
echo "Test year:    $TEST_YEAR"
echo "Output dir:   $TEST_OUTPUT_DIR"
echo "Start:        $(date)"
echo "========================================"

# Override ERA5_dat inline so test output goes to the test folder
ERA5_dat=$TEST_OUTPUT_DIR python "$SCRIPT_DIR/00_dwnld_arco_era5.py" --year "$TEST_YEAR"

EXIT_CODE=$?

echo "========================================"
echo "End:          $(date)"
echo "Exit code:    $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
