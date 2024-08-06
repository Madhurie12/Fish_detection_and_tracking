#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G          # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --account=rrg-whidden
#SBATCH --time=10:00:00    # Adjust time as needed
#SBATCH --output=exp_detection_fish_eval.out

# Activate the virtual environment
module load gcc/13.3.0
module load StdEnv/2020
module load python/3.10.2
virtualenv --no-download $SLURM_TMPDIR/yolov7_env
source $SLURM_TMPDIR/yolov7_env/bin/activate
pip install --no-index -r /home/madhurie/src/yolov7/requirements.txt

# Set experiment directory
EXP_DIR="/home/madhurie/scratch/fishdatasets"

# Path to the trained model weights
MODEL_WEIGHTS="$EXP_DIR/trained_models/exp_detection_fish_2epochs8/weights/best.pt"

# Path to the dataset configuration file
DATA_CONFIG="$EXP_DIR/yolov7-exp_detection_fish.yaml"

# Run evaluation
python /home/madhurie/src/yolov7/test.py \
    --weights $MODEL_WEIGHTS \
    --data $DATA_CONFIG \
    --img 640 \
    --batch-size 32 \
    --conf 0.001 \
    --iou-thres 0.6 \
    --task test \
    --device 0 \
    --save-json \
    --single-cls

echo "Evaluation completed."
