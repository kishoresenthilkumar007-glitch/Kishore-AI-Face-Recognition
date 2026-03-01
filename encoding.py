import cv2
import face_recognition
import os
import threading
import numpy as np
from scipy.spatial import distance as dist
from deepface import DeepFace

class KishoreAI:
    def __init__(self, database_path="Kishore_photos"):
        """Initialize system configurations and load face databases."""
        self.db_path = database_path
        self.ear_threshold = 0.21
        self.known_encodings = []
        self.known_names = []
        
        # State variables
        self.current_emotion = "Analyzing..."
        self.face_locations = []
        self.face_metadata = [] # Stores (name, eye_status)
        self.frame_count = 0
        
        self._prepare_database()
        self.video_capture = cv2.VideoCapture(0)

    def _prepare_database(self):
        """Loads and encodes reference images from the local directory."""
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
            print(f"[!] Warning: '{self.db_path}' created. Add images to proceed.")
            return

        print("[*] Loading known identities...")
        for file in os.listdir(self.db_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img = face_recognition.load_image_file(os.path.join(self.db_path, file))
                enc = face_recognition.face_encodings(img)
                if enc:
                    self.known_encodings.append(enc[0])
                    self.known_names.append("Kishore")
        print(f"[*] Database ready. Loaded {len(self.known_encodings)} signatures.")

    @staticmethod
    def calculate_ear(eye_landmarks):
        """Computes the Eye Aspect Ratio (EAR) to detect blink/closed status."""
        v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (v1 + v2) / (2.0 * h)

    def _async_emotion_analysis(self, frame_roi):
        """Internal method to process DeepFace analysis in a background thread."""
        try:
            res = DeepFace.analyze(frame_roi, actions=['emotion'], 
                                  enforce_detection=False, silent=True)
            self.current_emotion = res[0]['dominant_emotion'].capitalize()
        except Exception:
            self.current_emotion = "Unknown"

    def process_frame(self, frame):
        """Primary logic for detection, recognition, and UI rendering."""
        # AI Optimization: Process heavy logic every 5 frames
        if self.frame_count % 5 == 0:
            # Downscale for performance
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # 1. Identity & Landmark Detection
            self.face_locations = face_recognition.face_locations(rgb_small)
            encodings = face_recognition.face_encodings(rgb_small, self.face_locations)
            landmarks = face_recognition.face_landmarks(frame)

            # 2. Trigger Background Emotion Analysis
            threading.Thread(target=self._async_emotion_analysis, args=(small_frame,), daemon=True).start()

            # 3. Validation Logic
            self.face_metadata = []
            for i, enc in enumerate(encodings):
                matches = face_recognition.compare_faces(self.known_encodings, enc, tolerance=0.5)
                name = "Kishore" if True in matches else "Unknown"

                eye_status = "Open"
                if i < len(landmarks):
                    ear = (self.calculate_ear(landmarks[i]['left_eye']) + 
                           self.calculate_ear(landmarks[i]['right_eye'])) / 2.0
                    eye_status = "Open" if ear > self.ear_threshold else "Closed"
                
                self.face_metadata.append((name, eye_status))

        self._render_ui(frame)
        self.frame_count += 1

    def _render_ui(self, frame):
        """Handles drawing overlays and status alerts on the video feed."""
        for (t, r, b, l), (name, eye_status) in zip(self.face_locations, self.face_metadata):
            # Upscale coordinates back to original frame size
            t *= 4; r *= 4; b *= 4; l *= 4

            # Red Alert logic: Fails if Unknown, Eyes Closed
            is_valid = (name == "Kishore" and eye_status == "Open")
            color = (0, 255, 0) if is_valid else (0, 0, 255)

            # Draw bounding box and status label
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            label = f"{name} | Eyes: {eye_status} | Emotion: {self.current_emotion}"
            cv2.rectangle(frame, (l, t - 35), (r, t), color, cv2.FILLED)
            cv2.putText(frame, label, (l + 5, t - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        """Main loop to execute the monitoring application."""
        print("[*] Starting AI Stream...")
        while True:
            ret, frame = self.video_capture.read()
            if not ret: break

            self.process_frame(frame)
            cv2.imshow('Kishore AI Professional', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = KishoreAI()
    app.run()