import cv2
import numpy as np
from handTrackingModule import handTracker
import random
class DrawingBoard:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.board = np.zeros((720,1280,3), dtype=np.uint8)
        self.colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        self.current_color = self.colors[0]
        self.handTracker = handTracker()

    def start(self):
        cap = cv2.VideoCapture(0)

        while True:
            success, frame = cap.read()
            if not success:
                break

            frame = self.handTracker.handsFinder(frame)
            landmarks = self.handTracker.positionFinder(frame)

            if landmarks:
                self.handle_gestures(landmarks)

            cv2.imshow("Drawing Board", self.board)
            cv2.imshow("Webcam", frame)
            cv2.waitKey(1)

    def handle_gestures(self, landmarks):
        fingers = self.handTracker.fingersUp(landmarks)

        x, y = landmarks[8][1:]  # 8 is the tip of the index finger

        if fingers[1] and fingers[2]:  # If index and middle fingers are up
            cv2.circle(self.board, (x, y), 15, self.current_color, cv2.FILLED)
        elif fingers[1] and not any(fingers[2:]):  # If only index finger is up
            self.current_color = random.choice(self.colors)  # Change color
        elif all(fingers):  # If all fingers are up
            self.board = np.zeros((720,1280,3), dtype=np.uint8)  # Clear the board

    def get_frame(self):
        success, frame = self.cap.read()
        if not success:
            return None, None

        hand_frame = self.handTracker.handsFinder(frame)
        landmarks = self.handTracker.positionFinder(hand_frame)

        if landmarks:
            self.handle_gestures(landmarks)

        game_frame = self.board

        return game_frame, hand_frame

if __name__ == "__main__":
    DrawingBoard().start()