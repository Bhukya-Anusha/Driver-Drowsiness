import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import threading
from playsound import playsound

# Face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [78, 308, 13, 14]  # left, right, top, bottom

# Thresholds and counters
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6
CLOSED_EYE_FRAME_LIMIT = 20
eye_closed_counter = 0

# Alarm control
alarm_playing = False
alarm_lock = threading.Lock()

def play_alarm():
    global alarm_playing
    with alarm_lock:
        if not alarm_playing:
            alarm_playing = True
            try:
                playsound("alarm.wav")
            except Exception as e:
                print(f"Error playing alarm: {e}")
            alarm_playing = False

def get_landmark_coords(landmarks, indices, w, h):
    return [np.array([int(landmarks[i].x * w), int(landmarks[i].y * h)]) for i in indices]

def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth):
    A = distance.euclidean(mouth[2], mouth[3])
    C = distance.euclidean(mouth[0], mouth[1])
    return A / C

def run_detection():
    global eye_closed_counter, alarm_playing
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                left_eye = get_landmark_coords(landmarks, LEFT_EYE, w, h)
                right_eye = get_landmark_coords(landmarks, RIGHT_EYE, w, h)
                mouth = get_landmark_coords(landmarks, MOUTH, w, h)

                ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
                mar = calculate_mar(mouth)

                if ear < EAR_THRESHOLD:
                    eye_closed_counter += 1
                    if eye_closed_counter >= CLOSED_EYE_FRAME_LIMIT:
                        cv2.putText(frame, "DROWSINESS ALERT!", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        if not alarm_playing:
                            threading.Thread(target=play_alarm, daemon=True).start()
                else:
                    eye_closed_counter = 0
                    alarm_playing = False  # Reset alarm flag when eyes are open

                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "YAWNING DETECTED", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

                cv2.putText(frame, f"EAR: {ear:.2f}", (500, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (500, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start the detection
run_detection()
