#!/bin/bash
#SBATCH --job-name=single_label_all_thresholds
#SBATCH --account=r00018
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/all_single_label_threshold_%j.out
#SBATCH --error=logs/all_single_label_threshold_%j.err

echo "--------------------------------------"
echo "Loading conda environment..."
module load conda
conda activate HSD

echo "Environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python3)"
echo "--------------------------------------"

cd /geode2/home/u080/saashr/BigRed200/hsd-gab
echo "Working directory: $(pwd)"
echo "--------------------------------------"

echo "Running threshold script on ALL single-label files..."
echo "--------------------------------------"

# LOOP OVER ALL 3 DATA DIRECTORIES
for DIR in data/sing_label_data data/sing_label_data2_1 data/sing_label_data3_1; do
    echo "Processing directory: $DIR"
    echo "--------------------------------------"

    # LOOP OVER EVERY .npy FILE IN THE DIRECTORY
    for FILE in "$DIR"/*.npy; do
        echo ">>> Running scheme-2-main.py on: $FILE"
        python3 utils/scheme-2-main.py --file "$FILE"
        echo "--------------------------------------"
    done
done

echo "--------------------------------------"
echo "ALL SINGLE-LABEL THRESHOLD RUNS COMPLETE"
echo "--------------------------------------"

