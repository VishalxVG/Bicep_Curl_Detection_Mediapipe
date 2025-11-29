import cv2
import mediapipe as mp
import numpy as np


# --- 1. SMOOTHING CLASS ---
# This filters out the "jitter" when you move closer/further
class AngleSmoother:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_angle = None

    def smooth(self, new_angle):
        if self.prev_angle is None:
            self.prev_angle = new_angle
            return new_angle
        # Exponential Moving Average
        smoothed_angle = (self.alpha * new_angle) + ((1 - self.alpha) * self.prev_angle)
        self.prev_angle = smoothed_angle
        return smoothed_angle


# Initialize Smoother (Alpha 0.2 means trust new data 20%, old data 80% - very smooth)
smoother = AngleSmoother(alpha=0.2)


def calculate_angle_3d(a, b, c):
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist

    # Create vectors: Elbow -> Shoulder, Elbow -> Wrist
    ba = a - b
    bc = c - b

    # Calculate cosine angle
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

counter = 0
stage = "neutral"  # Initial state

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            world_landmarks = results.pose_world_landmarks.landmark
            landmarks = results.pose_landmarks.landmark

            # --- Extract Coordinates ---
            shoulder_3d = [
                world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                world_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z,
            ]
            elbow_3d = [
                world_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                world_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                world_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].z,
            ]
            wrist_3d = [
                world_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                world_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                world_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z,
            ]

            # --- Calculate & Smooth Angle ---
            raw_angle = calculate_angle_3d(shoulder_3d, elbow_3d, wrist_3d)
            angle = smoother.smooth(raw_angle)

            # --- Visualization Points (2D) ---
            elbow_2d = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            h, w = image.shape[:2]
            elbow_px = tuple(np.multiply(elbow_2d, [w, h]).astype(int))

            # --- DEBUGGING: Print the angle to the console ---
            # This is crucial. Watch your terminal to see what the numbers actually are.
            print(f"Angle: {int(angle)} | Stage: {stage}")

            cv2.putText(
                image,
                str(int(angle)),
                elbow_px,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if angle > 140:
                stage = "down"

            if angle < 80 and stage == "down":
                stage = "up"
                counter += 1

        except Exception as e:
            pass

        # --- Draw UI ---
        cv2.rectangle(image, (0, 0), (350, 100), (245, 117, 16), -1)

        cv2.putText(
            image,
            "REPS",
            (15, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(counter),
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            "STAGE",
            (100, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            stage,
            (100, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        cv2.imshow("Mediapipe Feed", image)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
