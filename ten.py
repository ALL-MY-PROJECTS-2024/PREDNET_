import os
import time
import numpy as np
import cv2
from keras.models import model_from_json
from prednet import PredNet
from data_utils import SequenceGenerator

# 스트리밍 URL
hls_url = "https://safecity.busan.go.kr/playlist/cnRzcDovL2d1ZXN0Omd1ZXN0QDEwLjEuMjEwLjIxMTo1NTQvdXM2NzZyM0RMY0RuczYwdE1ESXhMVEk9/index.m3u8"

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(hls_url)

# FPS 확인
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

# 목표 FPS (30 FPS)
target_fps = 30
frame_interval = int(fps / target_fps)
if frame_interval == 0:
    frame_interval = 1  # 최소 1로 설정하여 ZeroDivisionError 방지

# 비디오 송출용 설정 (실시간 송출을 위한 /stream)
stream_url = 'udp://127.0.0.1:5000'  # OpenCV를 사용한 스트리밍
stream2_url = 'udp://127.0.0.1:5001'  # 예측된 10분 후 영상을 위한 스트리밍

# PredNet 모델 로드
# 모델 아키텍처 로드
with open('prednet_kitti_model.json', 'r') as json_file:
    model_json = json_file.read()
prednet_model = model_from_json(model_json)
prednet_model.load_weights('prednet_kitti_weights.hdf5')

# 예측할 시간 프레임을 설정 (예: t+5 예측)
nt = 10  # 사용할 타임스텝 수 (프레임 수)
input_shape = (nt, 128, 160, 3)  # 예시 입력 크기
prednet = PredNet(stack_sizes=(3, 48, 96, 192), R_stack_sizes=(48, 96, 192), A_filt_sizes=(3, 3, 3), Ahat_filt_sizes=(3, 3, 3, 3), R_filt_sizes=(3, 3, 3, 3), output_mode='prediction')

# 영상 프레임 송출 및 예측 처리
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read frame.")
        break

    # 실시간 영상 송출 /stream
    if frame_count % frame_interval == 0:
        frame_resized = cv2.resize(frame, (160, 128))  # 예시로 160x128로 리사이즈
        cv2.imshow('Video Stream', frame_resized)
        
        # /stream 송출 (비디오 스트리밍)
        # 비디오 스트림 송출
        stream_out = cv2.VideoWriter(stream_url, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_resized.shape[1], frame_resized.shape[0]))
        stream_out.write(frame_resized)

    # 10분 후 예측 영상 송출 /stream2
    if frame_count >= nt:
        # 예측을 위한 입력 준비 (단일 프레임을 시퀀스 형태로 준비)
        input_sequence = np.array([frame_resized] * nt)  # 예시로 같은 프레임을 사용
        predictions = prednet.predict(input_sequence)

        # 예측 프레임을 /stream2로 송출
        predicted_frame = predictions[-1]  # 예측된 마지막 프레임 (t+5 예측)
        predicted_frame = np.uint8(predicted_frame)

        # /stream2 송출
        stream_out2 = cv2.VideoWriter(stream2_url, cv2.VideoWriter_fourcc(*'XVID'), fps, (predicted_frame.shape[1], predicted_frame.shape[0]))
        stream_out2.write(predicted_frame)

    frame_count += 1

    # 'q' 키를 누르면 종료
    key = cv2.waitKey(1)  # 실시간 영상에 대해 1ms 대기
    if key == ord('q'):
        print("Exiting...")
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
