import queue
import threading
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import torch
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Landmarks cho ROI
LEFT_EYE = [33, 133, 160, 159, 158, 153, 144, 145, 153, 163, 7]
RIGHT_EYE = [362, 263, 387, 386, 385, 380, 373, 374, 380, 390, 249]
MOUTH = [61, 291, 78, 308, 14, 13, 82, 312, 87, 317, 81, 311]


def extract_roi(frame, face_landmarks, landmark_ids, pad=5, draw=False, color=(0,255,0)):
    """Crop ROI từ các landmark"""
    ih, iw, _ = frame.shape
    points = [(int(face_landmarks.landmark[id].x * iw),
               int(face_landmarks.landmark[id].y * ih)) for id in landmark_ids]

    x_min = max(0, min([p[0] for p in points]) - pad)
    y_min = max(0, min([p[1] for p in points]) - pad)
    x_max = min(iw, max([p[0] for p in points]) + pad)
    y_max = min(ih, max([p[1] for p in points]) + pad)

    if draw:
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)


    return frame[y_min:y_max, x_min:x_max]


class DrowsinessDetector(QMainWindow):
    def __init__(self):
        super().__init__()

        # Trạng thái
        self.yawn_state = ''
        self.left_eye_state = ''
        self.right_eye_state = ''
        self.alert_text = ''

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0

        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False

        # Confidence
        self.left_conf = 0.0
        self.right_conf = 0.0
        self.yawn_conf = 0.0

        # Hiển thịROI
        self.show_roi = False

        # MediaPipe FaceMesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # GUI
        self.setWindowTitle("Somnolence Detection")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("background-color: white;")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 2px solid black;")
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        self.info_label = QLabel()
        self.info_label.setStyleSheet(
            "background-color: white; border: 1px solid black; padding: 10px;")
        self.layout.addWidget(self.info_label)

        # Nút bật/tắt ROI
        self.roi_button = QPushButton("Show ROI", self)
        self.roi_button.setCheckable(True)
        self.roi_button.clicked.connect(self.toggle_roi)
        self.layout.addWidget(self.roi_button)

        self.update_info()

        # Load YOLO models
        self.detectyawn = YOLO(
            r"real-time-drowsy-driving-detection\runs\detectyawn\train\weights\best.pt")
        self.detecteye = YOLO(
            r"real-time-drowsy-driving-detection\runs\detecteye\train\weights\best.pt")
        self.detectyawn.to(DEVICE)
        self.detecteye.to(DEVICE)

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        time.sleep(1.0)

        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

        # Threads
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.process_thread = threading.Thread(target=self.process_frames, daemon=True)

        self.capture_thread.start()
        self.process_thread.start()

    def toggle_roi(self):
        self.show_roi = self.roi_button.isChecked()
        if self.show_roi:
            self.roi_button.setText("Hide ROI")
        else:
            self.roi_button.setText("Show ROI")


    def update_info(self):
        if round(self.yawn_duration, 2) > 7.0:
            self.alert_text = "<p style='color: orange; font-weight: bold;'>⚠️ Alert: Prolonged Yawn!</p>"

        if round(self.microsleeps, 2) > 4.0:
            self.alert_text = "<p style='color: red; font-weight: bold;'>⚠️ Alert: Microsleep!</p>"

        info_text = (
            f"<div style='font-family: Arial, sans-serif; color: #333;'>"
            f"<h2 style='text-align: center; color: #4CAF50;'>Drowsiness Detector</h2>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"{self.alert_text}"
            f"<p><b>👁️ Blinks:</b> {self.blinks}</p>"
            f"<p><b>💤 Microsleeps:</b> {round(self.microsleeps, 2)} sec</p>"
            f"<p><b>😮 Yawns:</b> {self.yawns}</p>"
            f"<p><b>⏳ Yawn Duration:</b> {round(self.yawn_duration, 2)} sec</p>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"<p><b>👁️ Left Eye:</b> {self.left_eye_state} ({self.left_conf:.2f})</p>"
            f"<p><b>👁️ Right Eye:</b> {self.right_eye_state} ({self.right_conf:.2f})</p>"
            f"<p><b>👄 Mouth:</b> {self.yawn_state} ({self.yawn_conf:.2f})</p>"
            f"</div>"
        )
        self.info_label.setText(info_text)

    # Predict mắt
    def predict_eye(self, eye_frame, eye_state):
        results_eye = self.detecteye.predict(eye_frame, device=DEVICE, verbose=False)
        boxes = results_eye[0].boxes
        if len(boxes) == 0:
            return eye_state, 0.0

        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])
        conf = float(confidences[max_confidence_index])

        if class_id == 1:
            eye_state = "Close Eye"
        elif class_id == 0 and conf > 0.30:
            eye_state = "Open Eye"

        return eye_state, conf

    # Predict ngáp
    def predict_yawn(self, yawn_frame, face_landmarks, frame_shape):
        results_yawn = self.detectyawn.predict(yawn_frame, device=DEVICE, verbose=False)
        boxes = results_yawn[0].boxes
        yawn_yolo_state = "No Yawn"
        conf = 0.0
        if len(boxes) != 0:
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()
            max_confidence_index = np.argmax(confidences)
            class_id = int(class_ids[max_confidence_index])
            conf = float(confidences[max_confidence_index])
            yawn_yolo_state = "Yawn" if class_id == 0 else "No Yawn"

        # MAR tính thêm để tăng chắc chắn
        ih, iw, _ = frame_shape
        pts = [(face_landmarks.landmark[id].x * iw, face_landmarks.landmark[id].y * ih) for id in MOUTH]
        pts = np.array(pts)
        A = np.linalg.norm(pts[4] - pts[5])  # chiều cao
        B = np.linalg.norm(pts[0] - pts[1])  # chiều rộng
        MAR = A / B if B != 0 else 0

        if yawn_yolo_state == "Yawn" and MAR > 0.6:
            self.yawn_state = "Yawn"
        else:
            self.yawn_state = "No Yawn"

        return self.yawn_state, conf

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 1:
                    self.frame_queue.put(frame)
            else:
                break

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # ROI
                        left_eye_roi = extract_roi(frame, face_landmarks, LEFT_EYE, draw=self.show_roi, color=(0,255,0))
                        right_eye_roi = extract_roi(frame, face_landmarks, RIGHT_EYE, draw=self.show_roi, color=(255,0,0))
                        mouth_roi = extract_roi(frame, face_landmarks, MOUTH, draw=self.show_roi, color=(0,0,255))


                        try:
                            self.left_eye_state, self.left_conf = self.predict_eye(left_eye_roi, self.left_eye_state)
                            self.right_eye_state, self.right_conf = self.predict_eye(right_eye_roi, self.right_eye_state)
                            self.yawn_state, self.yawn_conf = self.predict_yawn(mouth_roi, face_landmarks, frame.shape)
                        except Exception as e:
                            print(f"Prediction error: {e}")

                        # Logic nhắm mắt
                        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
                            if not self.left_eye_still_closed:
                                self.left_eye_still_closed = True
                                self.blinks += 1
                            self.microsleeps += 1 / self.cap.get(cv2.CAP_PROP_FPS)
                            if self.microsleeps >= 2.0:
                                self.alert_text = "<p style='color: red; font-weight: bold;'>⚠️ Eyes closed >2s!</p>"
                                self.play_sound_in_thread()
                        else:
                            self.left_eye_still_closed = False
                            self.microsleeps = 0

                        # Logic ngáp
                        if self.yawn_state == "Yawn":
                            if not self.yawn_in_progress:
                                self.yawn_in_progress = True
                                self.yawn_duration = 0
                            self.yawn_duration += 1 / self.cap.get(cv2.CAP_PROP_FPS)
                        else:
                            if self.yawn_in_progress:
                                if self.yawn_duration > 0.5:  # tránh false positive
                                    self.yawns += 1
                                self.yawn_in_progress = False
                                self.yawn_duration = 0

                        self.update_info()
                        self.display_frame(frame)

            except queue.Empty:
                continue

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def play_alert_sound(self):
        winsound.Beep(1000, 500)

    def play_sound_in_thread(self):
        threading.Thread(target=self.play_alert_sound, daemon=True).start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())
