#!/bin/bash
#SBATCH --job-name=stitchwise_trial
#SBATCH --account=eecs504s001w26_class
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=%x-%j.log

# Load modules and activate environment
module purge
module load python3.10-anaconda

cd ~/StitchWise
source venv/bin/activate

# Run the Python script
python scripts/train_trial.py
