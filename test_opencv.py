from flask import Flask, Response
import cv2
import subprocess
import numpy as np

app = Flask(__name__)

# HLS 스트리밍 URL
video_stream_url = "https://safecity.busan.go.kr/playlist/cnRzcDovL2d1ZXN0Omd1ZXN0QDEwLjEuMjEwLjIxMTo1NTQvdXM2NzZyM0RMY0RuczYwdE1ESXhMVEk9/index.m3u8"

# FFmpeg로 비디오 캡처 객체 생성
def generate_frames():
    # FFmpeg 명령어로 HLS 비디오 스트림을 MJPEG 포맷으로 변환
    process = subprocess.Popen(
        ['ffmpeg', '-i', video_stream_url, '-f', 'mjpeg', '-'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    while True:
        # FFmpeg로 1프레임씩 읽기
        in_bytes = process.stdout.read(1024*1024)  # 읽을 바이트 크기 설정
        if not in_bytes:
            break

        # JPEG 이미지로 변환하여 스트리밍 형식으로 반환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + in_bytes + b'\r\n\r\n')

@app.route('/stream')
def stream():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
