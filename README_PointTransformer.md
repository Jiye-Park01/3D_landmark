# Point Transformer를 사용한 3D Face Landmark Detection

이 프로젝트는 Point Transformer 백본을 사용하여 3D 얼굴 랜드마크 검출을 수행합니다.

## 주요 특징

- **Backbone**: Pretrained Point Transformer (freeze → unfreeze 전략 사용)
- **Prediction Head**: Heatmap Regression
- **Post-processing**: Local Surface Unfolding & Registration

## 모델 구조

### 1. Point Transformer Backbone
- Pretrained autoencoder 가중치 사용
- Encoder만 사용하여 feature extraction
- Freeze → Unfreeze 전략으로 fine-tuning

### 2. Heatmap Head
- Point Transformer의 출력 feature를 입력으로 받음
- 3D Gaussian heatmap 생성
- 각 랜드마크별로 별도의 heatmap 출력

### 3. Training Strategy
- **Phase 1 (Freeze)**: 처음 10 에포크 동안 backbone을 freeze하고 heatmap head만 학습 (A.npy 파일들만 사용)
- **Phase 2 (Unfreeze)**: 10 에포크 후 backbone을 unfreeze하여 전체 모델 fine-tuning (모든 파일 타입 사용)

## 설치 및 설정

### 1. 의존성 설치
```bash
pip install torch torchvision
pip install numpy scipy
pip install tensorboard
```

### 2. Point Transformer 모듈 경로 설정
Point Transformer 모듈이 `/home/jhrew/jiye/3D_pointtransformer/model/pointtransformer/` 경로에 있어야 합니다.

### 3. Pretrained 가중치 확인
다음 경로에 pretrained 가중치 파일이 있어야 합니다:
```
/home/jhrew/jiye/3D_pointtransformer/autoencoder_pointTransformer/pointtransformer_autoencoder/model/model_best.pth
```

## 데이터셋 준비

데이터셋은 다음과 같은 구조로 준비되어야 합니다:

```
dataset/
├── shapes/
│   ├── train/
│   │   ├── shape_001.ply
│   │   ├── shape_002.ply
│   │   └── ...
│   └── val/
│       ├── shape_101.ply
│       ├── shape_102.ply
│       └── ...
└── landmarks/
    ├── train/
    │   ├── shape_001.txt
    │   ├── shape_002.txt
    │   └── ...
    └── val/
        ├── shape_101.txt
        ├── shape_102.txt
        └── ...
```

## 학습 실행

### 1. 기본 학습 스크립트 사용
```bash
./train_pointtransformer.sh
```

### 2. 직접 실행
```bash
python train3.py \
    --exp_name "PointTransformer_Landmark_Detection" \
    --data_dir "./dataset" \
    --dataset "face_landmark" \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.0001 \
    --num_points 2048 \
    --num_landmarks 56 \
    --freeze_epochs 10 \
    --unfreeze_epoch 10 \
    --pretrained_path "/home/jhrew/jiye/3D_pointtransformer/autoencoder_pointTransformer/pointtransformer_autoencoder/model/model_best.pth" \
    --cuda \
    --workers 4 \
    --sigma 5.0 \
    --position_loss_weight 0.3 \
    --landmark_range_penalty_weight 0.01
```

## 주요 파라미터

- `--exp_name`: 실험 이름
- `--data_dir`: 데이터셋 경로
- `--batch_size`: 배치 크기 (기본값: 32)
- `--epochs`: 총 학습 에포크 수 (기본값: 100)
- `--lr`: 학습률 (기본값: 0.0001)
- `--num_points`: 각 샘플당 포인트 수 (기본값: 2048)
- `--num_landmarks`: 랜드마크 수 (기본값: 56)
- `--freeze_epochs`: backbone을 freeze할 에포크 수 (기본값: 10)
- `--unfreeze_epoch`: backbone을 unfreeze할 에포크 (기본값: 10)
- `--sigma`: Gaussian heatmap의 표준편차 (기본값: 5.0)
- `--position_loss_weight`: 위치 손실 가중치 (기본값: 0.3)
- `--landmark_range_penalty_weight`: 랜드마크 범위 페널티 가중치 (기본값: 0.01)

## 모델 파일

### 수정된 파일들
1. **PAConv_model.py**: PAConv를 Point Transformer로 변경
2. **train3.py**: Pretrained 가중치 로딩 및 freeze → unfreeze 전략 구현
3. **My_args.py**: Point Transformer 관련 인자 추가

### 새로운 파일들
1. **train_pointtransformer.sh**: 학습 스크립트
2. **README_PointTransformer.md**: 이 README 파일

## 학습 과정

### Phase 1: Freeze Phase (Epoch 1-10)
- Point Transformer backbone을 freeze
- Heatmap head만 학습
- A.npy 파일들만 사용하여 초기 학습 안정화
- Pretrained feature를 활용하여 효율적인 학습

### Phase 2: Unfreeze Phase (Epoch 11+)
- Backbone을 unfreeze
- 모든 파일 타입 (A, B, C, F, K) 사용
- 전체 모델 fine-tuning
- 더 정교한 feature 학습

## 손실 함수

1. **Heatmap Loss**: Adaptive Wing Loss를 사용한 heatmap 예측 손실
2. **Position Loss**: 예측된 랜드마크와 실제 랜드마크 간의 L2 거리
3. **Range Penalty Loss**: 랜드마크가 유효한 범위 내에 있는지 확인하는 페널티

## 결과 저장

학습 결과는 다음 경로에 저장됩니다:
```
./checkpoints/{exp_name}/{dataset}/
├── models/
│   └── best_model.t7
└── training_log.txt
```

## 주의사항

1. Point Transformer 모듈의 경로가 올바르게 설정되어 있는지 확인
2. Pretrained 가중치 파일이 존재하는지 확인
3. GPU 메모리 부족 시 batch_size를 줄이거나 num_points를 줄임
4. 데이터셋 경로가 올바르게 설정되어 있는지 확인

## 문제 해결

### 1. Import 오류
Point Transformer 모듈을 찾을 수 없는 경우:
```python
import sys
sys.path.append('/home/jhrew/jiye/3D_pointtransformer/model/pointtransformer')
```

### 2. 메모리 부족
- batch_size를 줄이거나
- num_points를 줄이거나
- workers 수를 줄임

### 3. 가중치 로딩 오류
- Pretrained 가중치 파일 경로 확인
- 파일 형식 확인 (model_state_dict, state_dict 등) 