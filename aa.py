import os
from keras.models import model_from_json
from prednet import PredNet  # PredNet 클래스를 임포트
import numpy as np
import cv2
import time
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from threading import Thread, Event
from keras import backend as K
import tensorflow as tf


# TensorFlow 2.x의 즉시 실행 모드를 비활성화 (TensorFlow 1.x 호환성 유지)
tf.compat.v1.disable_eager_execution()
# Flask 애플리케이션 인스턴스를 생성
app = Flask(__name__)
CORS(app)  # CORS 활성화


# 'data' 폴더가 없으면 생성
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)



# # 학습데이터 생성
# def create_kitti_data_05():

# def create_kitti_data_10():

# def create_kitti_data_30():

    

# PredNet 모델을 로드하는 함수
def load_prednet_model():
    """PredNet 모델 로드 (실제 모델 파일 경로에 맞게 수정 필요)"""
    with open('model_data_keras2/prednet_kitti_model.json', 'r') as json_file:
        model_json = json_file.read()

    # 텐서플로우 세션 설정 전에 모델을 로드
    session = tf.compat.v1.Session()
    K.set_session(session)

    model = model_from_json(model_json, custom_objects={'PredNet': PredNet})
    model.load_weights('model_data_keras2/prednet_kitti_weights.hdf5')

    # 모델과 세션 연결
    K.set_session(session)

    return model

    
# 예측 함수 (t+N 예측)
def predict_at_tn(model, frame, nt=10, steps_ahead=1):
    """t+N 예측"""
    prediction = np.repeat(np.expand_dims(frame, axis=0), nt, axis=0)  # (1, nt, H, W, C) 형태로 확장
    for _ in range(steps_ahead):  # steps_ahead만큼 예측을 진행
        prediction_input = np.expand_dims(prediction, axis=0)  # (1, nt, H, W, C) 형태로
        prediction = model.predict(prediction_input)[0]  # 예측된 프레임을 업데이트
    return prediction  # t+N 예측 결과 반환




# 실시간 영상 스트리밍 처리 함수
save_frame_05m = []
save_frame_10m = []
save_frame_30m = []

def process_stream(source, stop_event, model):
    """현재 영상 스트리밍 연결, 읽기, 전송"""
    cap = None
    attempts = 0
    fps = 30  # 기본 FPS
    last_frame_time = time.time()
    frames_buffer = []  # 최근 프레임을 저장할 리스트
    frame_count = 0  # 저장된 프레임 카운트

    try:
        while not stop_event.is_set():
            if cap is None or not cap.isOpened():
                if attempts >= 5:
                    raise RuntimeError(f"Failed to connect to stream {source} after 5 attempts.")
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    print(f"Stream connected successfully: {source}")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    attempts = 0
                else:
                    print(f"Failed to connect to stream {source}. Attempt {attempts + 1}/5")
                    attempts += 1
                    time.sleep(3)
                    continue

            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {source}. Reconnecting...")
                cap.release()
                cap = None
                continue

            # 현재 영상 송출
            current_frame = frame.copy()

            # 입력 크기 맞추기 (리사이즈)
            resized_frame = cv2.resize(current_frame, (224, 224))  # PredNet 모델이 기대하는 입력 크기로 리사이즈
            resized_frame = resized_frame.astype(np.float32) / 255.0  # 정규화

            # 버퍼에 프레임 추가
            frames_buffer.append(resized_frame)
            frame_count += 1

            # 5분 (300프레임) 단위 저장
           
            
            if len(save_frame_05m) <= 300:  # 최대 300개만 저장
                save_frame_05m.append(current_frame)
                
            if len(save_frame_10m) <= 600:  
                save_frame_10m.append(current_frame)

            if len(save_frame_30m) <= 1800: 
                save_frame_05m.append(current_frame)


            # FPS 제한
            current_time = time.time()
            frame_interval = 1.0 / fps
            if current_time - last_frame_time < frame_interval:
                time.sleep(frame_interval - (current_time - last_frame_time))
                continue
            last_frame_time = current_time

            # 프레임을 JPEG로 인코딩
            _, buffer_current = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            current_frame_bytes = buffer_current.tobytes()

            # 현재 영상 스트리밍
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_frame_bytes + b'\r\n')

    except Exception as e:
        print(f"Error in process_stream for {source}: {e}")
    finally:
        if cap:
            cap.release()
        print(f"Stopped processing stream for {source}.")



# 예측된 영상 스트리밍 처리 함수
def process_stream2(source, stop_event, model):
    """예측된 영상 스트리밍 연결, 읽기, 전송"""
    cap = None
    attempts = 0
    fps = 30  # 기본 FPS
    last_frame_time = time.time()
    frames_buffer = save_frame_05m  # 최근 프레임을 저장할 리스트

    try:
        while not stop_event.is_set():
            if cap is None or not cap.isOpened():
                if attempts >= 5:
                    raise RuntimeError(f"Failed to connect to stream {source} after 5 attempts.")
                cap = cv2.VideoCapture(source)
                if cap.isOpened():
                    print(f"Stream connected successfully: {source}")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    attempts = 0
                else:
                    print(f"Failed to connect to stream {source}. Attempt {attempts + 1}/5")
                    attempts += 1
                    time.sleep(3)
                    continue

            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {source}. Reconnecting...")
                cap.release()
                cap = None
                continue

            current_frame = frame.copy()

            # 입력 크기 맞추기 (리사이즈)
            resized_frame = cv2.resize(current_frame, (224, 224))  # PredNet 모델이 기대하는 입력 크기로 리사이즈
            resized_frame = resized_frame.astype(np.float32) / 255.0  # 정규화

            # 예측된 프레임 계산
            prediction_input = np.expand_dims(np.array(frames_buffer), axis=0)
            frame_10min_predicted = model.predict(prediction_input)[0]  # 예측된 10분 뒤 프레임

            # FPS 제한
            current_time = time.time()
            frame_interval = 1.0 / fps
            if current_time - last_frame_time < frame_interval:
                time.sleep(frame_interval - (current_time - last_frame_time))
                continue
            last_frame_time = current_time

            # 예측된 프레임을 JPEG로 인코딩
            _, buffer_predicted = cv2.imencode('.jpg', frame_10min_predicted, [cv2.IMWRITE_JPEG_QUALITY, 80])
            predicted_frame_bytes = buffer_predicted.tobytes()

            # 예측된 영상 스트리밍
            yield (b'--frame2\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + predicted_frame_bytes + b'\r\n')

    except Exception as e:
        print(f"Error in process_stream2 for {source}: {e}")
    finally:
        if cap:
            cap.release()
        print(f"Stopped processing predicted stream for {source}.")


# 기존 스트림 안전 종료
def stop_existing_stream(source):
    """기존 스트림 안전 종료"""
    print(f"Stopping existing stream for {source}...")
    if source in stream_events:
        stop_event = stream_events[source]
        stop_event.set()  # 스트림 종료 신호 전달
        stream_threads[source].join()  # 스레드 종료 대기
        del stream_threads[source]
        del stream_events[source]
        print(f"Existing stream for {source} successfully stopped.")
    else:
        print(f"No active stream found for {source}.")


# 전역 변수로 스트림 관련 정보를 관리
stream_threads = {}
stream_events = {}


# frame 저장하기(5분단위, 10분단위, 30분단위)


# 스트리밍 요청 처리 (현재 영상)
@app.route('/stream1', methods=['GET'])
def stream_video():
    """스트림 요청 (현재 영상)"""
    source = request.args.get('rtspAddr')
    if not source:
        return jsonify({"error": "Missing 'rtspAddr' query parameter"}), 400

    # 기존 스트림 종료 처리
    if source in stream_threads:
        stop_existing_stream(source)

    # PredNet 모델 로드
    model = load_prednet_model()

    # 새 스트림 시작
    stop_event = Event()
    stream_events[source] = stop_event

    # 스레드 생성 및 시작
    stream_thread = Thread(target=process_stream, args=(source, stop_event, model))
    stream_threads[source] = stream_thread
    stream_thread.start()
    print(f"Stream from {source} started.")

    return Response(
        process_stream(source, stop_event, model),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )



# 예측 영상 스트리밍 요청 처리
@app.route('/stream2', methods=['GET'])
def stream_predicted_video():
    """스트림 요청 (예측 영상)"""
    source = request.args.get('rtspAddr')
    if not source:
        return jsonify({"error": "Missing 'rtspAddr' query parameter"}), 400

    # 기존 스트림 종료 처리
    if source in stream_threads:
        stop_existing_stream(source)

    # PredNet 모델 로드
    model = load_prednet_model()

    # 새 스트림 시작
    stop_event = Event()
    stream_events[source] = stop_event

    # 스레드 생성 및 시작
    stream_thread = Thread(target=process_stream2, args=(source, stop_event, model))
    stream_threads[source] = stream_thread
    stream_thread.start()
    print(f"Predicted stream from {source} started.")

    return Response(
        process_stream2(source, stop_event, model),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# 스트림 종료 요청 처리
@app.route('/stop-stream', methods=['POST'])
def stop_stream():
    """스트림 종료"""
    data = request.get_json()
    source = data.get('hlsAddr')
    if not source:
        return jsonify({"error": "Missing 'hlsAddr' parameter"}), 400

    stop_existing_stream(source)
    return jsonify({"message": f"Stream from {source} stopped."}), 200

# Flask 애플리케이션 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)