#!/usr/bin/bash
#SBATCH --job-name llm2colbert_training
#SBATCH --account fta-25-4
#SBATCH --partition qgpu
#SBATCH --time 8:00:00
#SBATCH --gpus 4 
#SBATCH --nodes 1

module --force purge
# Load stuff
ml purge
ml Anaconda3
source activate colbert

# Load .env file
set -a; source .env; set +a
wandb login

# experiment name
./run_training.sh 
