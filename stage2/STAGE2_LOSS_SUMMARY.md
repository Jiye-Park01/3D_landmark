# Stage 2 Loss Functions 정리

## 개요
Stage 2는 3D 얼굴 랜드마크 검출의 두 번째 단계로, 다음과 같은 4가지 주요 loss 함수를 사용합니다:

## 1. Heatmap Loss (히트맵 손실)
- **목적**: 예측된 3D 히트맵과 실제 히트맵 간의 차이를 최소화
- **구현**: MSE Loss
- **가중치**: 0.5
- **수식**: 
  ```
  L_heatmap = MSE(pred_heatmaps, true_heatmaps)
  ```

## 2. Regression Loss (회귀 손실)
- **목적**: 정제된 랜드마크 좌표와 실제 랜드마크 좌표 간의 차이를 최소화
- **구현**: MSE Loss
- **가중치**: 1.5 (가장 높은 가중치)
- **수식**: 
  ```
  L_regression = MSE(refined_landmarks, true_landmarks)
  ```

## 3. Unfolding Loss (전개 손실) - 선택적
- **목적**: 조악한 랜드마크와 정제된 랜드마크 간의 일관성 유지
- **구현**: L2 거리의 평균
- **가중치**: 0.1
- **활성화**: `--use_unfolding_loss` 플래그로 제어
- **수식**: 
  ```
  L_unfolding = mean(||refined_landmarks - coarse_landmarks||_2)
  ```

## 4. Registration Loss (등록 손실) - 선택적
- **목적**: 예측된 랜드마크와 실제 랜드마크 간의 정렬 오차 최소화
- **구현**: L2 거리의 평균
- **가중치**: 0.1
- **활성화**: `--use_registration_loss` 플래그로 제어
- **수식**: 
  ```
  L_registration = mean(||refined_landmarks - true_landmarks||_2)
  ```

## 총 손실 함수 (Total Loss)
```
L_total = 0.5 × L_heatmap + 1.5 × L_regression + 
          0.1 × L_unfolding (if enabled) + 0.1 × L_registration (if enabled)
```

## 평가 지표 (Evaluation Metrics)

### 1. Landmark Error
- **목적**: 랜드마크 예측 정확도 측정
- **구현**: 예측과 실제 랜드마크 간의 평균 L2 거리
- **수식**: 
  ```
  landmark_error = mean(||pred_landmarks - true_landmarks||_2)
  ```

### 2. NME (Normalized Mean Error)
- **목적**: 얼굴 크기로 정규화된 평균 오차
- **구현**: 얼굴 크기로 정규화된 L2 거리
- **수식**: 
  ```
  NME = mean(||pred_landmarks - true_landmarks||_2 / face_size)
  ```

## Loss 가중치 설정 근거

1. **Regression Loss (1.5)**: 가장 중요한 손실로, 최종 랜드마크 좌표의 정확성을 직접 측정
2. **Heatmap Loss (0.5)**: 중간 표현(히트맵)의 품질을 보장하지만 최종 목표보다는 낮은 가중치
3. **Unfolding Loss (0.1)**: 보조적 역할로 모델의 내부 일관성을 유지
4. **Registration Loss (0.1)**: 보조적 역할로 추가적인 정렬 제약 제공

## 조기 종료 (Early Stopping)
- **기준**: Regression Loss
- **Patience**: 20 에포크
- **Min Delta**: 1e-4
- **근거**: Regression Loss가 최종 성능을 가장 잘 나타내므로

## 사용 방법

### 기본 설정 (Heatmap + Regression Loss만 사용)
```bash
python train_stage2.py --data_dir ./data --stage1_model_path path/to/stage1/model
```

### 모든 Loss 사용
```bash
python train_stage2.py --data_dir ./data --stage1_model_path path/to/stage1/model --use_unfolding_loss --use_registration_loss
```

## 주의사항

1. **Registration Loss vs Regression Loss**: 
   - Registration Loss는 평가용 함수를 재사용하여 구현되었으나, 실제로는 Regression Loss와 동일한 계산을 수행
   - 중복성을 피하기 위해 일반적으로 둘 중 하나만 사용 권장

2. **가중치 조정**: 
   - 데이터셋과 모델에 따라 가중치 조정이 필요할 수 있음
   - 특히 Heatmap Loss와 Regression Loss의 스케일 차이를 고려해야 함

3. **메모리 사용량**: 
   - 모든 Loss를 동시에 사용하면 메모리 사용량이 증가
   - GPU 메모리가 부족한 경우 배치 크기 조정 필요