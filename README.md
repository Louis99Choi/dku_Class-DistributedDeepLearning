
# 단국대학교 인공지능융합학과 분산딥러닝 기말 프로젝트 Repository 입니다.

## 프로젝트 구조
```plaintext
├── requirements.txt # 가상환경에 설치되어야 할 라이브러리 
├── environments.yaml # 가상환경에 설치되어야 할 라이브러리
│
HRNet
├── tools 
│   ├── test.py # test 모델
│   ├── train_baseline.py #config(.yaml) 파일로 학습할 각 모델 정의 (pruning, quantize 등)
│   └── train.py # 본 프로젝트에서는 사용 안함
│
├── compare_model.py # 두 모델의 구조 및 크기 비교&시각화
├── compare_time.py # 두 모델의 추론 시간 비교&시각화
├── comparison_files # 비교 결과 이미지 저장
│
├── data # 데이터셋 경로
│   └── animal
│
├── experiments # 비교 실험 모델 config 파일 - Pruning 및 quantize 여부 & amount 명세
│   └── animal
│       ├── face_alignment_rat-base_hrnet_w18.yaml # base 모델
│       ├── face_alignment_rat-prune-02_hrnet_w18.yaml # Conv2d 레이어만 20% Pruning
│       ├── face_alignment_rat-prune-05_hrnet_w18.yaml # Conv2d 레이어만 50% Pruning
│       ├── face_alignment_rat-quantize-int8_hrnet_w18.yaml # Conv2d & Linear 레이어만 int8로 quantize
│       ├── face_alignment_rat-prune-05-quantize-int8_hrnet_w18.yaml # rat-prune-05 에서 int8로 quantize
│
├── log # TensorBoard Log 경로
│   └── ANIMAL
│       └── hrnet
│           ├── face_alignment_rat-base_hrnet_w18_2024-12-23-16-12
│           ├── face_alignment_rat_hrnet_w18_2024-12-23-13-09-baseline
│           ├── face_alignment_rat-prune-02_hrnet_w18_2024-12-23-17-30
│           ├── face_alignment_rat-prune-02_hrnet_w18_2024-12-23-22-23
│           ├── face_alignment_rat-prune-05_hrnet_w18_2024-12-23-19-07
│           ├── face_alignment_rat-prune-05_hrnet_w18_2024-12-23-23-01
│           ├── face_alignment_rat-prune-05-quantize-int8_hrnet_w18_2024-12-23-21-04
│           ├── face_alignment_rat-prune-05-quantize-int8_hrnet_w18_2024-12-24-00-07
│           ├── face_alignment_rat-quantize-int8_hrnet_w18_2024-12-23-20-12
│           └── face_alignment_rat-quantize-int8_hrnet_w18_2024-12-23-23-25
├── model_structures # compare_model.py 에서 사용되는 모델 구조 파일(확장자 없이 사용)
│   ├── baseline_structure
│   ├── baseline_structure.png
│   ├── baseline_weight_distribution.png
│   ├── face_alignment_rat-prune-02_hrnet_w18
│   ├── face_alignment_rat-prune-02_hrnet_w18.png
│   ├── face_alignment_rat-prune-05_hrnet_w18
│   ├── face_alignment_rat-prune-05_hrnet_w18.png
│   ├── face_alignment_rat-prune-05-quantize-int8_hrnet_w18
│   ├── face_alignment_rat-prune-05-quantize-int8_hrnet_w18.png
│   ├── face_alignment_rat-quantize-int8_hrnet_w18
│   └── face_alignment_rat-quantize-int8_hrnet_w18.png
└── README.md
