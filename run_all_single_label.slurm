#!/bin/bash
#SBATCH --job-name=hsd_single_label
#SBATCH --account=r00018
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00            # <-- 4 days
#SBATCH --output=logs/single_label_%j.out
#SBATCH --error=logs/single_label_%j.err

echo "Loading conda..."
module load conda
conda activate HSD

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python: $(which python3)"
echo "--------------------------------"

cd /geode2/home/u080/saashr/BigRed200/hsd-gab

echo "Working directory: $(pwd)"
echo "--------------------------------"

python3 utils/main_single_label.py

echo "--------------------------------"
echo "SINGLE-LABEL GRID+THRESHOLD RUN COMPLETE"
echo "--------------------------------"

