#!/bin/bash -l
#SBATCH -J Probe
#SBATCH -D ./
#SBATCH -o ./vision.out.%A.%a    # %A = jobid, %a = array task id
#SBATCH -e ./vision.err.%A.%a
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=24-25         # <-- adjust range to match number of models below


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
  "google/vit-base-patch16-224"
  "google/vit-large-patch16-224"
  "openai/clip-vit-large-patch14"
  "openai/clip-vit-base-patch32"
  "facebook/vit-mae-base"
  "facebook/vit-mae-large"
  "geolocal/StreetCLIP"
  "facebook/dinov2-base"
  "facebook/dinov2-large"
  "facebook/dinov2-giant"
  "facebook/dinov2-small"
  "facebook/webssl-mae300m-full2b-224"
  "facebook/webssl-dino300m-full2b-224"
  "facebook/metaclip-b16-400m"
  "facebook/metaclip-b32-400m"
  "facebook/metaclip-l14-fullcc2.5b"
  "facebook/metaclip-b16-fullcc2.5b"
  "facebook/metaclip-b32-fullcc2.5b"
  "facebook/webssl-dino1b-full2b-224"
  "facebook/webssl-mae1b-full2b-224"
  "facebook/metaclip-l14-400m"
  "facebook/metaclip-h14-fullcc2.5b"
  "facebook/webssl-dino7b-full8b-224"
  "facebook/dinov3-vitb16-pretrain-lvd1689m"
  "facebook/dinov3-vitl16-pretrain-lvd1689m"
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




