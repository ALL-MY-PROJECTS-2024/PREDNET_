import os
import time
import hickle as hkl
import numpy as np
import cv2

# 스트리밍 URL
hls_url = "https://safecity.busan.go.kr/playlist/cnRzcDovLzEwLjEuMjEwLjE4MToxMDAyOC9kZWlk/index.m3u8"

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(hls_url)

# 비디오가 제대로 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open the video stream.")
    exit()

# FPS 확인
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

# FPS 값이 유효한지 확인 (0보다 클 경우만 처리)
if fps <= 0:
    print("Error: Invalid FPS value.")
    exit()

# 목표 FPS (30 FPS)
#target_fps = 30
target_fps = 30
# FPS가 30보다 낮을 경우, 영상 속도를 맞추기 위한 대기 시간 계산
frame_interval = int(fps / target_fps)
if frame_interval == 0:
    frame_interval = 1  # 최소 1로 설정하여 ZeroDivisionError 방지

# 프레임 카운터
frame_count = 0
saved_count = 0

# 비디오 ID (예시로 설정, 실제 상황에 맞게 변경해야 함)
video_id = "kitti_data"

# 첫 번째 프레임 읽기 (frame.shape 사용을 위해)
ret, frame = cap.read()
if not ret:
    print("Error: Failed to read the first frame.")
    exit()

# 디렉토리가 존재하지 않으면 생성
output_dir = './kitti_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 훈련 데이터와 검증 데이터를 구분하여 파일 생성
output_train_file = f'./kitti_data/X_test.hkl'  # hkl 파일로 저장
output_sources_train_file = f'./kitti_data/sources_test.hkl'  # hkl 파일로 저장

# 훈련 및 검증 데이터 저장을 위한 리스트 초기화
frames_data_train = []
frames_data_val = []
sources_data_train = []
sources_data_val = []

# 비디오에서 프레임 읽기
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to read frame.")
        break

    # 매 프레임을 30 FPS에 맞춰 저장
    if frame_count % frame_interval == 0:
        # 원본 프레임을 저장 (리사이즈)
        frame_resized = cv2.resize(frame, (224, 224))  # (160, 128) 크기로 리사이즈

        frame_data = np.array(frame_resized)

        # 데이터가 훈련에 사용할 것인지 검증에 사용할 것인지 나누기
        if frame_count % 2 == 0:  # 짝수 프레임을 훈련 데이터로 저장
            frames_data_train.append(frame_data)
            sources_data_train.append(video_id.encode('utf-8'))  # 비디오 ID 저장
        else:  # 홀수 프레임을 검증 데이터로 저장
            frames_data_val.append(frame_data)
            sources_data_val.append(video_id.encode('utf-8'))  # 비디오 ID 저장

        saved_count += 1

        # 타임스탬프 생성
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

        # 콘솔에 출력
        print(f"{timestamp} - Saved frame {saved_count} to training and validation files.")

    # 프레임 카운터 증가
    frame_count += 1

    # 화면에 프레임 표시 (GUI 창을 띄워서 'q' 키 처리)
    cv2.imshow('Video Stream', frame_resized)

    # 'q' 키가 눌리면 종료
    key = cv2.waitKey(int(1000 / target_fps))  # 목표 FPS(30fps)에 맞춰 33.33ms 대기
    if key == ord('q'):  # 'q' 키가 눌리면 종료
        print("Saving process stopped by user (q pressed).")
        break

# hickle로 훈련 데이터 저장
hkl.dump(np.array(frames_data_train), output_train_file)
hkl.dump(np.array(sources_data_train), output_sources_train_file)

# hickle로 검증 데이터 저장
hkl.dump(np.array(frames_data_val), output_val_file)  # 추가된 부분
hkl.dump(np.array(sources_data_val), output_sources_val_file)  # 추가된 부분

# 리소스 해제
cap.release()
cv2.destroyAllWindows()

# 데이터 저장 후 확인을 위한 출력
print(f"훈련 데이터 저장 완료: {output_train_file}")
print(f"검증 데이터 저장 완료: {output_val_file}")

# hkl 파일에서 데이터 확인
train_data = hkl.load(output_train_file)
val_data = hkl.load(output_val_file)

# 저장된 데이터의 크기 출력 (640x640 해상도 확인)
print(f"훈련 데이터 크기: {train_data.shape}")  # (샘플 수, 프레임 수, 높이, 너비, 채널)
print(f"검증 데이터 크기: {val_data.shape}")  # (샘플 수, 프레임 수, 높이, 너비, 채널)
