# Stage 1 vs Stage 2 Global Feature Extraction 비교

## **Stage 1: PointTransformerLandmark (Max Pooling + MLP)**

### **Global Feature Extraction 과정:**
```python
# 1. Local Features from PointTransformer
encoder_features = self.backbone([x_flat, x_flat, batch_offsets])  # (B*N, 32)
encoder_features = encoder_features.view(batch_size, num_points, -1)  # (B, N, 32)
encoder_features = encoder_features.permute(0, 2, 1)  # (B, 32, N)

# 2. Global Feature Extraction - MAX POOLING
global_features = torch.max(encoder_features, dim=2, keepdim=True)[0]  # (B, 32, 1)
global_features = global_features.squeeze(-1)  # (B, 32)

# 3. Global Feature Projection - MLP
global_proj = self.global_proj(global_features)  # (B, 256)
# self.global_proj = nn.Sequential(
#     nn.Linear(32, 512),
#     nn.ReLU(inplace=True),
#     nn.Linear(512, 256),
#     nn.ReLU(inplace=True)
# )

# 4. Feature Expansion and Fusion
global_proj_expanded = global_proj.unsqueeze(-1).expand(-1, -1, num_points)  # (B, 256, N)
combined_features = torch.cat([encoder_features, global_proj_expanded], dim=1)  # (B, 288, N)
```

### **Stage 1의 특징:**
- **Max Pooling**: `torch.max(encoder_features, dim=2)` - 각 채널에서 최대값 선택
- **MLP Projection**: 32 → 512 → 256 차원 변환
- **Rich Features**: 256차원의 의미있는 전역 특징 생성
- **Feature Fusion**: Local(32) + Global(256) = Combined(288)

## **Stage 2: PointTransformerStage2 (Simple Averaging)**

### **Global Feature Extraction 과정:**
```python
# 1. Get heatmaps from Stage 1
heatmaps = self.stage1_model(x)  # (B, 68, N)

# 2. Global Feature Extraction - SIMPLE AVERAGING
# Step 1: Point-wise averaging
global_features = torch.mean(heatmaps, dim=2)  # (B, 68, N) → (B, 68)

# Step 2: Landmark-wise averaging  
global_features = torch.mean(global_features, dim=1, keepdim=True)  # (B, 68) → (B, 1)

# Step 3: Dimension expansion (placeholder)
global_features = global_features.expand(-1, 512)  # (B, 1) → (B, 512)
```

### **Stage 2의 특징:**
- **Simple Averaging**: `torch.mean()` - 단순 평균으로 정보 손실
- **No MLP**: 특징 변환 없이 단순 확장
- **Placeholder Features**: 1→512 확장이 의미없음
- **Information Loss**: 히트맵의 세부 정보 손실

## **두 단계의 차이점 비교**

### **1. 입력 데이터**
```
Stage 1: encoder_features (B, 32, N) ← PointTransformer 출력
Stage 2: heatmaps (B, 68, N) ← Stage 1 히트맵 출력
```

### **2. Global Feature Extraction 방법**
```
Stage 1: Max Pooling + MLP
├── Max Pooling: 각 채널의 최대값 선택 (정보 보존)
├── MLP: 32 → 512 → 256 (의미있는 특징 변환)
└── 결과: (B, 256) - 풍부한 전역 특징

Stage 2: Simple Averaging + Expansion
├── Point-wise averaging: 포인트별 평균 (정보 손실)
├── Landmark-wise averaging: 랜드마크별 평균 (더 많은 정보 손실)
├── Dimension expansion: 1 → 512 (의미없는 확장)
└── 결과: (B, 512) - placeholder 특징
```

### **3. 특징의 품질**
```
Stage 1 Global Features:
├── 품질: 높음 (Max Pooling + MLP)
├── 차원: 256 (의미있는 특징)
├── 정보 보존: 우수
└── 학습 가능: MLP로 학습된 특징

Stage 2 Global Features:
├── 품질: 낮음 (Simple Averaging)
├── 차원: 512 (placeholder)
├── 정보 보존: 열악
└── 학습 가능: 단순 확장으로 학습 불가
```

## **Stage 2 개선 방향**

### **방법 1: Stage 1의 Global Features 직접 활용**
```python
# Stage 1 모델 수정하여 global features 반환
def forward(self, x):
    # ... existing code ...
    
    # Return both heatmap and global features
    return heatmap, global_proj  # (B, 68, N), (B, 256)

# Stage 2에서 활용
def forward(self, x):
    heatmaps, stage1_global = self.stage1_model(x)
    
    # Use Stage 1's rich global features directly
    global_features = stage1_global  # (B, 256)
    
    # Project to required dimension if needed
    if global_features.size(1) != 512:
        global_features = self.global_proj(global_features)  # (B, 256) → (B, 512)
    
    return heatmaps, coarse_landmarks, refined_landmarks
```

### **방법 2: Stage 1의 중간 특징 활용**
```python
# Stage 1에서 중간 특징들 반환
def get_intermediate_features(self, x):
    # ... existing code until global feature extraction ...
    
    return {
        'local': encoder_features,      # (B, 32, N)
        'global': global_proj,         # (B, 256)
        'combined': combined_features,  # (B, 288, N)
        'heatmap': heatmap             # (B, 68, N)
    }

# Stage 2에서 활용
def forward(self, x):
    features = self.stage1_model.get_intermediate_features(x)
    
    # Use Stage 1's rich global features
    global_features = features['global']  # (B, 256)
    
    # Project to required dimension
    global_features = self.global_proj(global_features)  # (B, 256) → (B, 512)
    
    return features['heatmap'], coarse_landmarks, refined_landmarks
```

## **핵심 결론**

1. **Stage 1**: **Max Pooling + MLP**로 품질 높은 전역 특징 생성
2. **Stage 2**: **Simple Averaging**으로 정보 손실이 많은 특징 생성
3. **개선 방향**: Stage 1의 품질 높은 전역 특징을 Stage 2에서 직접 활용
4. **현재 구현**: Stage 2의 Global Feature Extraction이 비효율적

**Stage 2는 Stage 1의 히트맵만 받아서 단순 평균으로 특징을 추출하는 대신, Stage 1의 품질 높은 전역 특징을 직접 활용하는 것이 훨씬 효율적입니다!** 