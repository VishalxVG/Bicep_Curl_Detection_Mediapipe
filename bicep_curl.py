import cv2  # The "Eyes" (OpenCV)
import mediapipe as mp  # The "Brain" (MediaPipe)
import numpy as np  # The "Calculator" (Math)


def calculate_angle(a, b, c):
    a = np.array(a)  # Shoulder
    b = np.array(b)  # Elbow
    c = np.array(c)  # Wrist

    # Calculate the angle using arctan2 (a fancy math function to find angles)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# Setup MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Tool to draw the skeleton lines
mp_pose = mp.solutions.pose  # The pose detection model

# Start the video feed (0 usually means your main webcam)
cap = cv2.VideoCapture(0)

# Set webcam resolution (example: 1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variables to count reps
counter = 0
stage = None  # "Up" or "Down"

# Start the Pose detection loop
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()  # Read one frame (picture) from the camera

        # 1. Color Adjustment
        # OpenCV sees colors as Blue-Green-Red (BGR).
        # MediaPipe likes Red-Green-Blue (RGB). We convert it.
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # Lock the image to make processing faster

        # 2. The Magic Detection
        results = pose.process(image)  # This line finds the skeleton!

        # 3. Convert color back to BGR so we can see it normally
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 4. Extract Landmarks (The Dots)
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates of the Left Arm dots
            shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            # Calculate the angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # Convert normalized elbow coords → pixel coords
            h, w = image.shape[:2]
            elbow_px = tuple(np.multiply(elbow, [w, h]).astype(int))

            # Visualize angle on screen
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

            # 5. Curl Counter Logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter += 1
                print(f"Reps: {counter}")

        except:
            pass  # If we can't see the person, just skip this frame

        # 6. Draw the Score Box
        cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)

        # Rep Data
        cv2.putText(
            image,
            "REPS",
            (15, 12),  # Coordinates (x , y) x from top , y from left
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

        # Stage Data (Up or Down)
        cv2.putText(
            image,
            "STAGE",
            (80, 12),
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

        # 7. Draw the Skeleton on top of the video
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # Show the video on screen
        cv2.imshow("Mediapipe Feed", image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to Quit
            break

    cap.release()
    cv2.destroyAllWindows()
