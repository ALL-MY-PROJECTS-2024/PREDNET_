import cv2
import time
import hickle as hkl
import numpy as np
import os

# 스트리밍 URL
hls_url = "https://safecity.busan.go.kr/playlist/cnRzcDovL2d1ZXN0Omd1ZXN0QDEwLjEuMjEwLjIxMTo1NTQvdXM2NzZyM0RMY0RuczYwdE1ESXhMVEk9/index.m3u8"

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
target_fps = 30

# 프레임 간격 계산 (30 FPS로 설정)
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


frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # 흑백처리
frame_resized = cv2.resize(frame, (320, 320))  # (width, height) 순서

# 디렉토리가 존재하지 않으면 생성
output_dir = './kitti_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 훈련 데이터와 검증 데이터를 구분하여 파일 생성
output_train_file = f'./kitti_data/X_train.hkl'  # hkl 파일로 저장
output_val_file = f'./kitti_data/X_val.hkl'  # 검증용 파일 추가
output_sources_train_file = f'./kitti_data/sources_train.hkl'  # hkl 파일로 저장
output_sources_val_file = f'./kitti_data/sources_val.hkl'  # 검증용 소스 파일 추가

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

    # FPS 30에 맞춰 프레임을 저장
    if frame_count % frame_interval == 0:
        # 이미지 데이터를 numpy 배열로 변환 후 저장
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
    key = cv2.waitKey(1)  # 1ms 대기
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
