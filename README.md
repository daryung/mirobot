YOLO 결과
│
├── Mask 1
│    ├── Contour 1 → Path 1
│    └── Contour 2 → Path 2
│
├── Mask 2
│    └── Contour 1 → Path 3
│
└── Mask 3
     └── Contour 1 → Path 4

image_to_strokes(image)

이미지를 받아 윤곽선 기반 stroke 목록으로 변환
이미지 크기 조정
grayscale + Gaussian blur + Canny edge detection
contour 추출
너무 짧은 contour 제거
점 간 간격 제한으로 중복점 제거
전체 contour를 canvas 크기 기준으로 정규화
stroke 간격 최소화 및 최대 point 수 제한
최종적으로 optimize_stroke_order()로 시퀀스 정리해서 반환
“사진을 그림 선으로 변환하는 핵심 함수”
optimize_stroke_order(strokes)

여러 stroke들의 순서를 자연스럽게 정렬
현재 마지막 점 기준으로 가장 가까운 다음 stroke를 선택
필요한 경우 stroke를 reverse해서 연결
로봇이 그리기 시작점에서 이어서 그릴 수 있게 정렬하는 함수
의미: 로봇이 불필요하게 점프하지 않도록 최적화