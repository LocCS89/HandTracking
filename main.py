import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, request
import cv2
from handTrackingModule import handTracker
from game import DrawingBoard
app = Flask(__name__)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
tracker = handTracker()
game=DrawingBoard()
def generate_video_stream():
    cap = cv2.VideoCapture(0)
    while True:
        success, image = cap.read()
        if not success:
            break
        image = tracker.handsFinder(image)
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
def gen():
    while True:
        game_frame, hand_frame = game.get_frame()
        print("Game Frame Shape:", game_frame.shape if game_frame is not None else "None")
        print("Hand Frame Shape:", hand_frame.shape if hand_frame is not None else "None")

        if game_frame is None or hand_frame is None:
            print('No frame available')
            continue

        hand_frame = cv2.resize(hand_frame, (game_frame.shape[1], game_frame.shape[0]))  # Add this line

        combined_frame = np.concatenate((game_frame, hand_frame), axis=1)
        print("Combined Frame Shape:", combined_frame.shape)

        ret, jpeg = cv2.imencode('.jpg', combined_frame)
        frame = jpeg.tobytes()
        print('Sending frame')
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/handTracking')
def hand_tracking():
    return render_template('handTracking.html')

@app.route('/dashboard')
def index():
    return render_template('dashboard.html')
@app.route('/game_feed')
def game_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/game')
def draw():
    return render_template('game.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def generate():
    cap = cv2.VideoCapture(0)
    while True:
        success, image = cap.read()
        if not success:
            break
        print("Before:", image.shape)  # Add this line
        image = tracker.handsFinder(image)
        print("After:", image.shape)  # And this line
        ret, jpeg = cv2.imencode('.jpg', image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)