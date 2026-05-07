#!/bin/bash -l
#SBATCH -J FineTu
#SBATCH -D ./
#SBATCH -o ./ft.out.%A.%a    # %A = jobid, %a = array task id
#SBATCH -e ./ft.err.%A.%a
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --array=1-5         # <-- adjust range to match number of models below


# Usage check: pass experiment name as first argument when submitting:
# --- env setup (modify if your cluster uses a different conda/shell init) ---
module purge
module load anaconda/3/2023.03


conda activate geovit

# --- Model list ---
models=(
  "google/vit-large-patch16-224-in21k"
  "openai/clip-vit-large-patch14"
  "facebook/vit-mae-large"
  "facebook/dinov2-large"
  "facebook/dinov2-giant"
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

cd ../

for SEED in 1 2 3 4 5
do
  echo "Running with seed $SEED"
  python -u ./finetune.py --model "$MODEL" --seed "$SEED"
done

echo "Finished model: $MODEL"




