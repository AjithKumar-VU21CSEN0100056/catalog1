import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe for pose estimation
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load the hat image (ensure it has a transparent background or a solid background)
hat = cv2.imread('hat.png', cv2.IMREAD_UNCHANGED)

# Check if the image was loaded correctly
if hat is None:
    print("Error: 'hat1.png' image file not found or could not be loaded.")
    exit()

# Check if the image has an alpha channel (transparency)
if hat.shape[2] == 4:
    # Image has an alpha channel
    hat_bgr = hat[:, :, :3]
    hat_alpha = hat[:, :, 3] / 255.0
else:
    # Image does not have an alpha channel, create a mask of ones
    hat_bgr = hat
    hat_alpha = np.ones(hat_bgr.shape[:2], dtype=hat_bgr.dtype)

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay `img_overlay` on `img` at the position `pos` and blend using `alpha_mask`."""
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if no part of the image is in the frame
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]

    img_crop[:] = alpha * img_overlay_crop + (1 - alpha) * img_crop

    return img

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Perform pose detection
        results = pose.process(image)

        # Convert the image color back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for overlay (in this case, for the hat)
            nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
            forehead_x = int(nose.x * frame.shape[1])
            forehead_y = int(nose.y * frame.shape[0] - 100)  # Adjust based on the hat size

            # Resize hat to fit the face size
            hat_width = int(2 * (landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x - landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x) * frame.shape[1])
            hat_height = int(hat_width * hat_bgr.shape[0] / hat_bgr.shape[1])
            resized_hat = cv2.resize(hat_bgr, (hat_width, hat_height), interpolation=cv2.INTER_AREA)
            resized_hat_alpha = cv2.resize(hat_alpha, (hat_width, hat_height), interpolation=cv2.INTER_AREA)

            # Overlay the hat on the frame
            frame = overlay_image_alpha(frame, resized_hat, (forehead_x - hat_width // 2, forehead_y - hat_height // 2), resized_hat_alpha)

        # Display the resulting frame
        cv2.imshow('AR Shopping - Virtual Try-On', frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
