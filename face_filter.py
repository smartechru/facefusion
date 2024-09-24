import cv2
import dlib
import numpy as np

# Load pre-trained face detector and facial landmarks predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./data/shape_predictor_68_face_landmarks.dat")


# Function to detect face
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None  # No face detected

    return faces[0]


# Function to check if face is frontal
def is_front_face(shape):
    left_eye = np.mean([shape.part(i).x for i in range(36, 42)])
    right_eye = np.mean([shape.part(i).x for i in range(42, 48)])
    nose = shape.part(30).x

    if abs(left_eye - right_eye) < 15 and abs(nose - (left_eye + right_eye) / 2) < 10:
        return True
    return False


# Function to detect if eyes are open
def are_eyes_open(shape):
    left_eye_height = np.mean(
        [shape.part(37).y - shape.part(41).y, shape.part(38).y - shape.part(40).y]
    )
    right_eye_height = np.mean(
        [shape.part(43).y - shape.part(47).y, shape.part(44).y - shape.part(46).y]
    )

    # Threshold to determine if eyes are open, adjust according to face size
    return left_eye_height > 4 and right_eye_height > 4


# Function to check if image is blurry
def is_blurry(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < 100  # Lower threshold means more blur


# Function to check brightness and contrast
def check_brightness_contrast(image, face):
    # Focus on the face region
    face_region = image[face.top() : face.bottom(), face.left() : face.right()]
    gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # Brightness: average pixel intensity
    brightness = np.mean(gray_face)

    # Contrast: standard deviation of pixel intensities
    contrast = np.std(gray_face)

    # Check if brightness and contrast fall within normal ranges
    brightness_ok = 100 <= brightness <= 180
    contrast_ok = contrast > 50

    return brightness_ok and contrast_ok


# Function to check skin tone
def is_skin_tone_normal(image, face):
    # Convert image to YCrCb color space (useful for skin detection)
    face_region = image[face.top() : face.bottom(), face.left() : face.right()]
    ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)
    # Define range of typical skin colors
    min_skin = np.array([0, 135, 85], dtype=np.uint8)
    max_skin = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, min_skin, max_skin)
    skin_area_ratio = np.sum(mask) / mask.size

    # At least 30% of the face region should be within the normal skin tone range
    return skin_area_ratio > 0.3


# Main function to detect a clear front face with normal features
def detect_clear_front_face(image_path):
    image = cv2.imread(image_path)
    face = detect_face(image)

    if face is None:
        return "No face detected"

    # Detect landmarks
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, face)

    if not is_front_face(landmarks):
        print("Face is not frontal")

    if not are_eyes_open(landmarks):
        print("Eyes are not clearly open")

    if is_blurry(image):
        print("Image is blurry")

    if not check_brightness_contrast(image, face):
        print("Brightness/contrast are not normal")

    if not is_skin_tone_normal(image, face):
        print("Face color tone is abnormal")

    # return "Clear front face detected"


def main():
    # Test the function
    dir_path = "D:/AI/Dataset/CelebDB/Bai Bai He_004195_F"
    # file_path = "Bai Bai He_B3mQ5c_20240617214811 (43).jpg"
    # file_path = "Bai Bai He_B3mQ5c_20240617214811 (45).jpg"
    # file_path = "Bai Bai He_B3mQ5c_20240617214811 (104).jpg"
    # file_path = "Bai Bai He_B3mQ5c_20240617214811 (91).jpg"
    # file_path = "Bai Bai He_B3mQ5c_20240617214811 (53).jpg"
    # file_path = "Bai Bai He_B3mQ5c_20240617214811 (4).jpg"
    # file_path = "Bai Bai He_B3mQ5c_20240617214811 (117).jpg"
    # file_path = "Bai Bai He_B3mQ5c_20240617214811 (99).jpg"
    # file_path = "Bai Bai He_B3mQ5c_20240617214811 (36).jpg"
    file_path = "Bai Bai He_B3mQ5c_20240617214811 (101).jpg"
    image_path = f"{dir_path}/{file_path}"
    result = detect_clear_front_face(image_path)
    print(result)


if __name__ == "__main__":
    main()
