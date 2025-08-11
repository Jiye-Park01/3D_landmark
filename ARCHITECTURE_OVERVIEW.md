# 3D Landmark Detection - Complete Architecture Overview

## 전체 아키텍처 구조

### 1. Stage 1: PointTransformerLandmark

```
Input: 3D Point Cloud (B, 3, N) 
       ↓
PointTransformer Backbone (Encoder Only)
       ↓
Feature Extraction: (B, 32, N) 
       ↓
Global Feature Projection: (B, 256)
       ↓
Feature Concatenation: Local(32) + Global(256) = (B, 288, N)
       ↓
HeatmapHead (Conv1D): (B, 288, N) → (B, 68, N)
       ↓
Output: Heatmap (B, 68, N)
```

### 2. Stage 2: PointTransformerStage2

```
Input: 3D Point Cloud (B, 3, N)
       ↓
Stage 1 Model (Inherited)
       ↓
Heatmap Prediction: (B, 68, N)
       ↓
Global Feature Extraction
       ↓
CoarseToFineHead (MLP)
       ├── Coarse MLP: Features → (B, 68, 3)
       └── Refinement MLP: Features + Coarse → (B, 68, 3)
       ↓
Output: Heatmap + Coarse + Refined Landmarks
```

## 상세 Feature 추출 과정

### PointTransformer Backbone
```
Input: (B, 3, N) → Reshape → (B*N, 3)
       ↓
PointTransformer Encoder
       ↓
Output: (B*N, 32) → Reshape → (B, N, 32) → Permute → (B, 32, N)
```

### Global Feature Processing
```
Local Features: (B, 32, N)
       ↓
Global Max Pooling: (B, 32, 1) → (B, 32)
       ↓
Global Projection MLP: 32 → 512 → 256
       ↓
Expand: (B, 256, N)
```

### Feature Fusion
```
Local Features: (B, 32, N)
Global Features: (B, 256, N)
       ↓
Concatenation: (B, 288, N)
```

## 히트맵 예측 과정

### HeatmapHead (Conv1D)
```
Input: (B, 288, N)
       ↓
Conv1D(288→512, k=1) + BN + ReLU
       ↓
Conv1D(512→256, k=1) + BN + ReLU  
       ↓
Conv1D(256→68, k=1)
       ↓
Output: (B, 68, N) - 각 포인트에 대한 랜드마크 확률
```

## 핵심 설계 원리

### 1. **Feature Hierarchy**
- **Local Features**: PointTransformer가 추출한 포인트별 특징 (32차원)
- **Global Features**: 전체 포인트 클라우드의 전역 정보 (256차원)
- **Combined Features**: Local + Global = 288차원

### 2. **Spatial vs Global Processing**
- **Conv1D**: 포인트별 공간적 특징 처리 (히트맵용)
- **MLP**: 전역 특징을 통한 좌표 직접 예측 (회귀용)

### 3. **Multi-scale Feature Fusion**
- **Point Level**: 각 포인트의 3D 위치 정보
- **Global Level**: 전체 얼굴의 구조적 정보
- **Combined**: 두 정보를 결합하여 정확한 랜드마크 위치 예측

## 데이터 흐름 요약

```
Point Cloud → PointTransformer → Local Features (32D)
                    ↓
              Global Pooling → Global Features (256D)
                    ↓
              Feature Fusion → Combined Features (288D)
                    ↓
              Conv1D Head → Heatmap (68 landmarks × N points)
                    ↓
              ArgMax → Landmark Coordinates (68 × 3)
```

이 구조의 핵심은 **PointTransformer로 공간적 특징을 추출하고, 전역 정보와 결합하여 각 포인트에 대한 랜드마크 확률을 예측**하는 것입니다. 