#!/bin/bash
#SBATCH -J HSD  # Job name
#SBATCH -p general                    # Partition name (use "general" or appropriate partition)
#SBATCH -o logs/baseline_initial_%j.txt    # Standard output file with job ID
#SBATCH -e logs/baseline_initial_%j.err    # Standard error file with job ID
#SBATCH --mail-type=ALL               # Email notifications for all job events
#SBATCH --mail-user=mealieff@iu.edu   # Email address for notifications
#SBATCH --nodelist=nid0212,nid0213,nid0215,nid0217,nid0218
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --time=2-02:00:00             # Maximum run time (2 days, 2 hours)
#SBATCH --mem=16G                     # Memory allocation (16 GB)
#SBATCH -A r00018                     # SLURM account name


# Load the virtual environment - everything's been updated there
module load conda

# Activate the ENV
conda activate HSD

# Set up the working directory
cd ~/hsd-gab/

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the Python script
#python3 utils/embedding-debugging.py
#python3 utils/svm.py
#python3 utils/get-embeddings.py   
#python3 utils/svm-postembedding.py
#python3 utils/svm-binary.py
#python3 utils/svm-multi2.py
#python3 utils/merged-resampler21.py
#python3 utils/merged-resampler31.py
#bash run_all_evaluations.sh 
#python3 utils/main2.py --data_dir baseline_data --setting baseline --confidence --labels 8 --split_dev --baseline_data_dir baseline_data
#bash multi_label_eval.sh 
#bash run_all_evaluations.sh
bash run_all_single_label.sh
