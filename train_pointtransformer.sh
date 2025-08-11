#!/bin/bash

# Point Transformer를 사용한 3D Face Landmark Detection 학습 스크립트

# 환경 설정
export CUDA_VISIBLE_DEVICES=0

# 학습 파라미터
DATA_DIR="./dataset"
EXP_NAME="PointTransformer_Landmark_Detection"
DATASET="face_landmark"
BATCH_SIZE=32
EPOCHS=100
LEARNING_RATE=0.0001
NUM_POINTS=2048
NUM_LANDMARKS=68
FREEZE_EPOCHS=10
UNFREEZE_EPOCH=10

# Pretrained 가중치 경로
PRETRAINED_PATH="/home/jhrew/jiye/3D_pointtransformer/autoencoder_pointTransformer/pointtransformer_autoencoder/model/model_best.pth"

echo "Starting Point Transformer training for 3D Face Landmark Detection"
echo "Data directory: $DATA_DIR"
echo "Experiment name: $EXP_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Number of points: $NUM_POINTS"
echo "Number of landmarks: $NUM_LANDMARKS"
echo "Freeze epochs: $FREEZE_EPOCHS (using A.npy files only)"
echo "Unfreeze epoch: $UNFREEZE_EPOCH (using all file types)"
echo "Pretrained path: $PRETRAINED_PATH"

# 학습 실행
python train3.py \
    --exp_name "$EXP_NAME" \
    --data_dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --num_points $NUM_POINTS \
    --num_landmarks $NUM_LANDMARKS \
    --freeze_epochs $FREEZE_EPOCHS \
    --unfreeze_epoch $UNFREEZE_EPOCH \
    --pretrained_path "$PRETRAINED_PATH" \
    --cuda \
    --workers 4 \
    --sigma 5.0 \
    --position_loss_weight 0.3 \
    --landmark_range_penalty_weight 0.01

echo "Training completed!" 