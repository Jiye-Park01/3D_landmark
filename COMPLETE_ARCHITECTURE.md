# 3D Landmark Detection - Complete Architecture with Freeze/Unfreeze Strategy

## 전체 학습 파이프라인 (Stage 1 + Stage 2)

### **Stage 1: PointTransformerLandmark with Freeze/Unfreeze**

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Start                           │
│              Load Pretrained PointTransformer              │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Phase 1: FREEZE Training                    │
│                    (Epochs 0 → unfreeze_epoch)            │
├─────────────────────────────────────────────────────────────┤
│ • Backbone: FROZEN ❄️ (requires_grad = False)             │
│ • Trainable: HeatmapHead ONLY                              │
│ • Gaussian Sigma: 1.5 (coarse heatmap)                    │
│ • Focus: Fast head adaptation                              │
│ • Dataset: A files only                                    │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Phase Transition                            │
│                (Epoch == unfreeze_epoch)                  │
├─────────────────────────────────────────────────────────────┤
│ • Unfreeze Backbone: model.unfreeze_backbone()            │
│ • Reset Early Stopping: patience_counter = 0              │
│ • Change Sigma: 1.5 → 1.0 (fine heatmap)                 │
│ • Reinitialize Optimizer: All parameters                  │
│ • Reset Best Loss: best_test_loss = inf                   │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Phase 2: UNFREEZE Training                  │
│                    (After unfreeze_epoch)                  │
├─────────────────────────────────────────────────────────────┤
│ • Backbone: TRAINABLE 🔥 (requires_grad = True)            │
│ • Trainable: Full Model (Backbone + Head)                 │
│ • Gaussian Sigma: 1.0 (precise heatmap)                   │
│ • Focus: Feature refinement + end-to-end optimization     │
│ • Dataset: Still A files only                              │
└─────────────────────────────────────────────────────────────┘
```

## **Stage 1 상세 아키텍처 (Freeze/Unfreeze 포함)**

### **Model Architecture**
```
Input: (B, 3, N) Point Cloud
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                PointTransformer Backbone                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Status: FREEZE/UNFREEZE controlled                  │   │
│  │ • Freeze: requires_grad = False                     │   │
│  │ • Unfreeze: requires_grad = True                    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
Local Features: (B, 32, N)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                Global Feature Processing                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Max Pooling → Global MLP (32→512→256)              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
Feature Fusion: Local(32) + Global(256) = (B, 288, N)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                HeatmapHead (Conv1D)                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Status: ALWAYS TRAINABLE                            │   │
│  │ • Conv1D(288→512→256→68)                           │   │
│  │ • BatchNorm + ReLU                                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
Output: Heatmap (B, 68, N)
```

### **Freeze/Unfreeze Implementation**
```python
def freeze_backbone(self):
    """Freeze backbone encoder"""
    for param in self.backbone.parameters():
        param.requires_grad = False
    print("Backbone encoder frozen")

def unfreeze_backbone(self):
    """Unfreeze backbone encoder"""
    for param in self.backbone.parameters():
        param.requires_grad = True
    print("Backbone encoder unfrozen")
```

## **Stage 2: PointTransformerStage2 (Stage 1 상속)**

### **Model Architecture**
```
Input: (B, 3, N) Point Cloud
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                Stage 1 Model (Inherited)                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Complete Stage 1 pipeline                         │   │
│  │ • HeatmapHead (Conv1D)                              │   │
│  │ • Freeze/Unfreeze status inherited                  │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
Heatmap: (B, 68, N)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                Global Feature Extraction                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • From heatmaps → Global features (B, 512)         │   │
│  │ • AdaptiveAvgPool1d(1)                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                CoarseToFineHead (MLP)                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Coarse MLP: (B, 512) → (B, 68, 3)                │   │
│  │ • Linear + ReLU + Dropout                          │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Refinement MLP: (B, 512+204) → (B, 68, 3)        │   │
│  │ • Input: Global features + Coarse coordinates      │   │
│  │ • Linear + ReLU + Dropout                          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
       │
       ▼
Output: Heatmap + Coarse + Refined Landmarks
```

## **Freeze/Unfreeze 전략의 핵심**

### **1. Phase 1: Freeze (안정적 초기 학습)**
```
Backbone: ❄️ FROZEN
├── PointTransformer weights 고정
├── 사전 훈련된 특징 보존
├── 빠른 헤드 적응
└── 과적합 방지

HeatmapHead: 🔥 TRAINABLE
├── Conv1D 층들 학습
├── 기본적인 히트맵 생성 학습
└── σ=1.5로 coarse learning
```

### **2. Phase Transition (학습 전략 전환)**
```
🔄 Critical Moment
├── Backbone unfreeze
├── Early stopping reset
├── Sigma change (1.5→1.0)
├── Optimizer reinitialize
└── Full model training 시작
```

### **3. Phase 2: Unfreeze (정밀한 최적화)**
```
Backbone: 🔥 TRAINABLE
├── PointTransformer fine-tuning
├── Task-specific 특징 학습
├── End-to-end 최적화
└── σ=1.0으로 precise learning

Full Model: 🔥 TRAINABLE
├── Backbone + Head 동시 학습
├── Feature refinement
├── 정확도 향상
└── 최종 성능 달성
```

## **학습 과정에서의 변화**

### **Loss Function (동일하게 유지)**
```
Total Loss = 1.0×Heatmap + 0.5×Unfolding + 0.3×Registration
```
- Freeze/Unfreeze 단계 모두 동일한 손실 함수 사용
- 가중치는 변하지 않음

### **Gaussian Sigma 변화**
```
Freeze Phase: σ = 1.5 (coarse, wide Gaussian)
Unfreeze Phase: σ = 1.0 (fine, narrow Gaussian)
```
- Coarse → Fine 학습 전략
- 점진적 정밀도 향상

### **Dataset Strategy**
```
Both Phases: A files only
├── 일관된 데이터셋 사용
├── 학습 안정성 확보
└── 비교 가능한 결과
```

## **전체 파이프라인 요약**

```
Stage 1: PointTransformerLandmark
├── Freeze Phase: Backbone 고정, Head 학습 (σ=1.5)
├── Unfreeze Phase: 전체 모델 학습 (σ=1.0)
└── Output: 정교한 히트맵

Stage 2: PointTransformerStage2  
├── Stage 1 모델 상속 (히트맵 + Freeze/Unfreeze 상태)
├── 추가 MLP 헤드 (Coarse + Refined)
└── Output: 히트맵 + 정밀한 랜드마크 좌표
```

이 Freeze/Unfreeze 전략은 **안정적인 초기 학습 → 정밀한 최적화**의 두 단계 접근법으로, 모델의 성능을 단계적으로 향상시키는 핵심 메커니즘입니다! 