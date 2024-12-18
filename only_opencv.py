import cv2
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
import time
from threading import Thread, Event
import numpy as np

# PredNet 모델을 로드하거나 다른 예측 모델을 사용하는 부분
# 예를 들어, 'keras'를 사용하여 모델을 로드할 수 있습니다.
# prednet_model = load_prednet_model()  # 예시로 모델 로드 코드 추가 (실제 구현 필요)

app = Flask(__name__)
CORS(app)

stream_threads = {}
stream_events = {}

# PredNet 예측 함수 (예시로 추정)
def predict_with_prednet(frame):
    """PredNet을 사용한 예측"""
    # frame은 (H, W, C) 형태로 들어옵니다.
    # 여기서 예측을 수행하고 예측 결과를 반환합니다.
    # 예시로, 프레임을 그대로 반환하는 부분은 실제 PredNet을 적용해야 합니다.
    # 예시로 예측된 결과를 추가로 적용하는 부분을 작성해야 합니다.
    
    # 모델 예측 코드 예시
    # prediction = prednet_model.predict(np.expand_dims(frame, axis=0))
    # 예측 결과 처리 후, 원본과 예측 결과를 나란히 표시하는 부분 처리
    # 예측된 프레임을 적절하게 가공하고 반환합니다.
    
    # 현재 예시로 예측된 결과가 없으므로, 프레임을 그대로 반환합니다.
    return frame  # 실제 예측이 이루어져야 합니다.

def process_stream(source, stop_event):
    """스트림 연결, 읽기, 전송"""
    cap = None
    attempts = 0
    fps = 30  # 기본 FPS
    last_frame_time = time.time()
    start_time = time.time()  # 스트림 시작 시간 기록
    prediction_delay = 600  # 10분 (600초) 후 예측 시작

    try:
        while not stop_event.is_set():
            # 스트림 연결 시도
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

            # 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from {source}. Reconnecting...")
                cap.release()
                cap = None
                continue

            # 예측을 10분 후부터 수행하도록 설정
            if time.time() - start_time > prediction_delay:
                frame = predict_with_prednet(frame)

            # FPS 제한
            current_time = time.time()
            frame_interval = 1.0 / fps
            if current_time - last_frame_time < frame_interval:
                time.sleep(frame_interval - (current_time - last_frame_time))
                continue
            last_frame_time = current_time

            # 프레임을 JPEG로 인코딩
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        print(f"Error in process_stream for {source}: {e}")
    finally:
        if cap:
            cap.release()
        print(f"Stopped processing stream for {source}.")

@app.route('/stream', methods=['GET'])
def stream_video():
    """스트림 요청"""
    source = request.args.get('hlsAddr')
    if not source:
        return jsonify({"error": "Missing 'hlsAddr' query parameter"}), 400

    # 기존 스트림 종료 처리
    if source in stream_threads:
        stop_existing_stream(source)

    # 새 스트림 시작
    stop_event = Event()
    stream_events[source] = stop_event

    # 스레드 생성 및 시작
    stream_thread = Thread(target=process_stream, args=(source, stop_event))
    stream_threads[source] = stream_thread
    stream_thread.start()
    print(f"Stream from {source} started.")

    return Response(
        process_stream(source, stop_event),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

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

@app.route('/stop-stream', methods=['POST'])
def stop_stream():
    """스트림 종료"""
    data = request.get_json()
    source = data.get('hlsAddr')
    if not source:
        return jsonify({"error": "Missing 'hlsAddr' parameter"}), 400

    stop_existing_stream(source)
    return jsonify({"message": f"Stream from {source} stopped."}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
