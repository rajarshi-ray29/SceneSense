#!/bin/bash
#SBATCH --job-name=a4_cse538   # Job name
#SBATCH --output=logs/slurm_%j.out    # Output log file
#SBATCH --error=logs/slurm_%j.err     # Error log file
#SBATCH --time=24:00:00               # Time limit (adjust as needed)
#SBATCH --partition=defq              # Use GPU partition (change if necessary)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --mem=32G                      # Memory allocation
#SBATCH --cpus-per-task=8              # Number of CPU cores
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH --nodes=1                      # Number of nodes

source /home/rarray/anaconda3/etc/profile.d/conda.sh

conda activate a4_cse538

echo "Starting Bimodal Training Pipeline..."
python main.py

conda deactivate

## format_data.py