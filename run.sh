#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request 4 CPUs
#$ -pe omp 4

# Request 1 GPU 
#$ -l gpus=2

# Specify the minimum GPU compute capability 
#$ -l gpu_c=3.5
#$ -l gpu_type=P100|V100

# Specify hard time limit
#$ -l h_rt=12:00:00

module load miniconda
# module load python3/3.8.6
module load pytorch/1.7.0
conda activate /projectnb/ivc-ml/dlteif/dlteif/venvs/py3.8
export LD_LIBRARY_PATH=/share/pkg.7/miniconda/4.9.2/install/lib/
JOB_ID="exp0"
DATASET=""
DATA_DIR=""
python main.py  --dataset=${DATASET} --data_dir=${DATA_DIR} \
                --job_id=${JOB_ID} --gpus=2 --dist --arch=CNN --loss=CE --balanced \
