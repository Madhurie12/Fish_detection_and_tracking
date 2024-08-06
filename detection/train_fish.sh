#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32G          # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --account=rrg-whidden
#SBATCH --time=54:00:00
#SBATCH --output=exp_detection_fish_train.out

# Activate the virtual environment
module load gcc/13.3.0
module load StdEnv/2020
module load python/3.10.2
virtualenv --no-download $SLURM_TMPDIR/yolov7_env
source $SLURM_TMPDIR/yolov7_env/bin/activate
pip install --no-index -r /home/madhurie/src/yolov7/requirements.txt

# Set experiment name and directory
EXP_NAME="exp_detection_fish"
EXP_DIR="/home/madhurie/scratch/fishdatasets"

# Train YOLOv7 model for 500 epochs
python /home/madhurie/src/yolov7/train.py \
    --single-cls \
    --workers 8 \
    --device 0 \
    --batch-size 32 \
    --data $EXP_DIR/yolov7-exp_detection_fish.yaml \
    --img 640 640 \
    --cfg /home/madhurie/src/yolov7/cfg/training/yolov7.yaml \
    --project $EXP_DIR/trained_models \
    --name ${EXP_NAME}_2epochs \
    --hyp /home/madhurie/src/yolov7/data/hyp.scratch.p5.yaml \
    --weights /home/madhurie/src/yolov7/yolov7.pt \
    --epochs 500 \
    --save_period 1
