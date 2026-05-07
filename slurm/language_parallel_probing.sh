#!/bin/bash -l
#SBATCH -J Probe
#SBATCH -D ./
#SBATCH -o ./language.out.%A.%a    # %A = jobid, %a = array task id
#SBATCH -e ./language.err.%A.%a
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=1-7         # <-- adjust range to match number of models below


# Usage check: pass experiment name as first argument when submitting:
if [ $# -lt 3 ]; then
  echo "Error: No experiment name provided."
  echo "Usage: sbatch run_probing_array.sh <experiment_name> <bias> <sampling>"
  exit 1
fi
EXPERIMENT_NAME=$1
BIAS=$2
SAMPLING=$3
PROBE=${4:-ridge}  # default to "ridge" if not provided

# --- env setup (modify if your cluster uses a different conda/shell init) ---
module purge
module load anaconda/3/2023.03


conda activate geovit

# --- Model list ---
models=(
  "llava-hf/llava-1.5-7b-hf"
  "Qwen/Qwen2.5-VL-3B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
  "google/gemma-3-4b-pt"
  "google/gemma-3-12b-pt"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
)

# Validate array index and pick model (SLURM_ARRAY_TASK_ID is 1..N)
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}
IDX=$((TASK_ID - 1))

if [ $IDX -lt 0 ] || [ $IDX -ge ${#models[@]} ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 2
fi

MODEL="${models[$IDX]}"
echo "Starting task $TASK_ID (array index $IDX) for model: $MODEL"
echo "Experiment: $EXPERIMENT_NAME"
echo "JobID: $SLURM_JOB_ID"

cd ../

# Run your programs (stdout/stderr also go to the SBATCH -o/-e files above)
python -u ./probe_landscapes.py --experiment "$EXPERIMENT_NAME" --model "$MODEL" --bias "$BIAS" --aggregation "$SAMPLING" -f "../vit/filtered_landmarks.hdf5" --probe "$PROBE"
python -u ./run_probing.py    --experiment "$EXPERIMENT_NAME" --model "$MODEL" --bias "$BIAS" --aggregation "$SAMPLING" -f "../vit/geocells_clusters.hdf5" --probe "$PROBE"

echo "Finished model: $MODEL"




