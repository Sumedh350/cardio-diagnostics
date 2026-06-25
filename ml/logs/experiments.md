# Experiment Log

All training runs are appended here automatically by `ml/heart/train.py` and `ml/lung/train.py`.

**Targets**: heart ≥ 85% | lung ≥ 82%

---

## 2026-06-24T04:11:01
- **Model**: heart  (attempt 1/4)
- **Architecture**: Conv2D(32,64,128) → Dense(256) → Dropout(0.5) → sigmoid
- **Optimizer**: adam  lr=0.001
- **Augmentation**: none
- **Epochs run**: 33
- **Train accuracy**: 0.9858
- **Val accuracy**: 0.9599
- **Target met**: ✓ yes  (target ≥ 0.85)
- **Next action**: target reached — done

## 2026-06-24T11:48:40
- **Model**: lung  (attempt 1/4)
- **Architecture**: Conv2D(32,64,128) → Dense(256) → Dropout(0.5) → softmax(4)
- **Optimizer**: adam  lr=0.001
- **Augmentation**: none
- **Epochs run**: 13
- **Train accuracy**: 0.3394
- **Val accuracy**: 0.5283
- **Target met**: ✗ no  (target ≥ 0.82)
- **Next action**: attempt 2: Conv2D(32,64,128,256) → Dense(256) → Dropout(0.5) → softmax(4)

## 2026-06-24T12:19:47
- **Model**: lung  (attempt 2/4)
- **Architecture**: Conv2D(32,64,128,256) → Dense(256) → Dropout(0.5) → softmax(4)
- **Optimizer**: adam  lr=0.0005
- **Augmentation**: none
- **Epochs run**: 8
- **Train accuracy**: 0.2741
- **Val accuracy**: 0.2714
- **Target met**: ✗ no  (target ≥ 0.82)
- **Next action**: attempt 3: Conv2D(32,64,128,256)+SpecAugment → Dense(256) → Dropout(0.5) → softma

## 2026-06-24T12:52:52
- **Model**: lung  (attempt 3/4)
- **Architecture**: Conv2D(32,64,128,256)+SpecAugment → Dense(256) → Dropout(0.5) → softmax(4)
- **Optimizer**: adam  lr=0.0003
- **Augmentation**: SpecAugment T=20 F=8
- **Epochs run**: 8
- **Train accuracy**: 0.1809
- **Val accuracy**: 0.0740
- **Target met**: ✗ no  (target ≥ 0.82)
- **Next action**: attempt 4: Conv2D(32,64,128,256)+SpecAugment+SGD(mom=0.9) → Dense(256) → softmax(

## 2026-06-24T14:12:39
- **Model**: lung  (attempt 4/4)
- **Architecture**: Conv2D(32,64,128,256)+SpecAugment+SGD(mom=0.9) → Dense(256) → softmax(4)
- **Optimizer**: sgd  lr=0.001
- **Augmentation**: SpecAugment T=20 F=8
- **Epochs run**: 21
- **Train accuracy**: 0.4041
- **Val accuracy**: 0.4978
- **Target met**: ✗ no  (target ≥ 0.82)
- **Next action**: all 4 attempts exhausted

## 2026-06-24T21:19:56
- **Model**: lung-binary  (attempt 1/4)
- **Architecture**: Conv2D(32,64,128) → GAP → Dense(256) → Dropout(0.5) → sigmoid
- **Optimizer**: adam  lr=0.001
- **Augmentation**: none
- **Epochs run**: 39
- **Train accuracy**: 0.8695
- **Val accuracy**: 0.7823
- **Target met**: ✗ no  (target ≥ 0.85)
- **Next action**: attempt 2: Conv2D(32,64,128,256) → GAP → Dense(256) → Dropout(0.5) → sigmoid

## 2026-06-24T22:46:10
- **Model**: lung-binary  (attempt 2/4)
- **Architecture**: Conv2D(32,64,128,256) → GAP → Dense(256) → Dropout(0.5) → sigmoid
- **Optimizer**: adam  lr=0.0005
- **Augmentation**: none
- **Epochs run**: 20
- **Train accuracy**: 0.8352
- **Val accuracy**: 0.7518
- **Target met**: ✗ no  (target ≥ 0.85)
- **Next action**: attempt 3: Conv2D(32,64,128,256)+SpecAugment → GAP → Dense(256) → Dropout(0.5) → 

## 2026-06-25T01:21:01
- **Model**: lung-fast  (attempt 1/4)
- **Architecture**: Conv2D(16,32,64x64) → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid
- **Optimizer**: adam  lr=0.001
- **Batch**: 64
- **Augmentation**: none
- **Epochs run**: 20
- **Train accuracy**: 0.6488
- **Val accuracy**: 0.6647
- **Target met**: ✗ no  (target ≥ 0.82)
- **Next action**: attempt 2: Conv2D(16,32,64x64) lr+50% → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid

## 2026-06-25T01:22:14
- **Model**: lung-fast  (attempt 2/4)
- **Architecture**: Conv2D(16,32,64x64) lr+50% → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid
- **Optimizer**: adam  lr=0.0015
- **Batch**: 64
- **Augmentation**: none
- **Epochs run**: 25
- **Train accuracy**: 0.6612
- **Val accuracy**: 0.6778
- **Target met**: ✗ no  (target ≥ 0.82)
- **Next action**: attempt 3: Conv2D(16,32,64,64x64) → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid

## 2026-06-25T01:23:06
- **Model**: lung-fast  (attempt 3/4)
- **Architecture**: Conv2D(16,32,64,64x64) → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid
- **Optimizer**: adam  lr=0.001
- **Batch**: 64
- **Augmentation**: none
- **Epochs run**: 15
- **Train accuracy**: 0.6666
- **Val accuracy**: 0.6851
- **Target met**: ✗ no  (target ≥ 0.82)
- **Next action**: attempt 4: Conv2D(16,32,64,64x64) batch=32 → MaxPool → GAP → Dense(64) → Dropout(0.3) → sig

## 2026-06-25T01:24:15
- **Model**: lung-fast  (attempt 4/4)
- **Architecture**: Conv2D(16,32,64,64x64) batch=32 → MaxPool → GAP → Dense(64) → Dropout(0.3) → sigmoid
- **Optimizer**: adam  lr=0.001
- **Batch**: 32
- **Augmentation**: none
- **Epochs run**: 19
- **Train accuracy**: 0.7197
- **Val accuracy**: 0.7054
- **Target met**: ✗ no  (target ≥ 0.82)
- **Next action**: all attempts exhausted

## 2026-06-25T03:41:46
- **Model**: lung-quick  (attempt 1)
- **Architecture**: Conv2D(32,64,128)+BN → GAP → Dense(256) → Dropout → sigmoid  [T=104]
- **Optimizer**: adam  lr=0.001  ReduceLROnPlateau
- **Dropout**: 0.5  L2=0e+00  T_crop=104
- **Epochs run**: 28
- **Train accuracy**: 0.8274
- **Val accuracy**: 0.7591
- **Target met**: ✓ yes  (target ≥ 0.75)
- **Next action**: target reached
