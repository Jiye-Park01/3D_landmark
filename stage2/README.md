# Stage 2: 다양한 표정 기반 정밀 3D 랜드마크 예측

## 🎯 프로젝트 개요

Stage 1에서 학습된 Point Transformer + Heatmap Head를 기반으로, 다양한 표정을 포함한 전체 메쉬 데이터(약 10,000개)에서 정밀한 3D 랜드마크 좌표를 예측하는 모델입니다.

## 🧠 모델 구조

```
입력: Point Cloud (B, N, 3)
↓
Backbone: Point Transformer (Stage 1 fine-tuned 가중치 로드)
↓
분기:
  1) Heatmap Head → (B, L, N)   : 각 랜드마크에 대한 3D heatmap 예측
  2) Coarse-to-Fine Head:
       - Coarse MLP → (B, L, 3) : 초기 좌표 예측
       - Refinement MLP → (B, L, 3) : 정밀 좌표 보정
출력:
  - heatmaps (B, L, N)
  - coarse_landmarks (B, L, 3)
  - refined_landmarks (B, L, 3)
```

## 📁 파일 구조

```
stage2/
├── PointTransformer_Stage2.py      # Stage 2 모델 정의
├── dataset_stage2.py               # Stage 2 데이터셋
├── train_stage2.py                 # Stage 2 학습 스크립트
├── inference_stage2.py             # Stage 2 추론 스크립트
├── checkpoints/                    # 모델 체크포인트 저장
├── results/                        # 추론 결과 저장
└── README.md                       # 이 파일
```

## 🧪 Loss 구성

```python
total_loss = (
    0.5 * heatmap_loss +       # 3D Gaussian MSE
    1.5 * regression_loss      # Final refined landmark 좌표와 GT 간의 L2
)
```

## 🚀 사용법

### 1. 데이터 준비

Stage 2용 데이터를 다음 구조로 준비하세요:

```
dataset_stage2/
├── points/
│   ├── sample_1.npy
│   ├── sample_2.npy
│   └── ...
└── landmarks/
    ├── sample_1.npy
    ├── sample_2.npy
    └── ...
```

- `points/`: 정규화된 3D point cloud (8192포인트)
- `landmarks/`: 각 메쉬의 정답 3D landmark 좌표

### 2. 학습 실행

```bash
cd /home/jhrew/jiye/3D_landmark/stage2

python train_stage2.py \
    --data_dir ./dataset_stage2 \
    --stage1_model_path ../checkpoints/PointTransformer_Landmark_Detection/custom/models/best_model.t7 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --sigma 1.0
```

### 3. 추론 실행

```bash
python inference_stage2.py \
    --model_path ./checkpoints/best_model_stage2.t7 \
    --data_dir ./dataset_stage2 \
    --num_samples 5 \
    --output_dir ./results
```

## 📊 평가 지표

- **Chamfer Distance (CD)**: 예측 landmark와 GT landmark 간의 평균 거리
- **Normalized Mean Error (NME)**: 얼굴 크기로 정규화한 landmark 오차
- **Coarse vs Refined 비교**: Coarse-to-Fine 학습의 효과 측정

## 🔧 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--epochs` | 100 | 학습 에포크 수 |
| `--batch_size` | 8 | 배치 크기 |
| `--lr` | 1e-3 | 학습률 |
| `--sigma` | 1.0 | Heatmap Gaussian sigma |
| `--num_points` | 8192 | 샘플링할 포인트 수 |

## 📈 학습 전략

1. **Stage 1 모델 로드**: fine-tuned된 Point Transformer backbone 사용
2. **Coarse-to-Fine 학습**: 초기 예측 → 정밀 보정의 2단계 학습
3. **Early Stopping**: Total Loss 기준으로 patience=10
4. **가중치 적용**: Heatmap Loss (0.5) + Regression Loss (1.5)

## 🎨 시각화 결과

추론 후 다음 파일들이 생성됩니다:

- `./results/heatmaps/`: 3D heatmap 시각화
- `./results/visualizations/`: 3D landmark 예측 결과
- `./results/sample_*.npy`: 상세 결과 데이터

## 🔍 모델 특징

1. **Heatmap + Regression 동시 학습**: 3D 분포와 정확한 좌표를 모두 예측
2. **Coarse-to-Fine 구조**: 초기 예측 → 정밀 보정으로 정확도 향상
3. **Stage 1 가중치 활용**: 사전 학습된 backbone의 지식 전이
4. **다양한 표정 대응**: 10,000개 메쉬 데이터로 일반화 성능 향상

## ⚠️ 주의사항

1. **Stage 1 모델 경로**: 반드시 유효한 Stage 1 모델 경로를 지정해야 합니다.
2. **데이터 형식**: 포인트 클라우드와 랜드마크의 좌표계가 일치해야 합니다.
3. **메모리 사용량**: 8192 포인트 × 배치 크기에 따라 GPU 메모리 사용량이 증가할 수 있습니다.

## 📞 문의

문제가 발생하거나 추가 기능이 필요하시면 언제든 문의해 주세요! 