import os
import cv2
import dlib
import math
import numpy as np
from datetime import datetime


def log(text, level=0, tag="MAIN", filename="log.txt"):
    """
    Record device execution log to the file or console
    :param text: source text
    :param level: log level
    :param tag: log tag
    :param filename: output log file name
    """
    # build the final text to print
    final_text = f"{2 * level * ' '}"
    if tag is not None:
        final_text = f"{final_text}[{tag.upper()}] "

    # add timestamp
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    final_text = f"[{timestamp}] {final_text}{text}"

    # print log to console
    print(final_text)

    # write file
    if filename is not None:
        ftr = open(filename, "a")
        ftr.write(f"{final_text}\n")
        ftr.close()


class FaceShapeClassifier:
    """
    HEART: The heart face shape has a fairly wide forehead, and a narrow chin.
    ROUND: The round face shape has a wide hairline and fullness below the cheeks.
    SQUARE: The square face shape has both a wide hairline and jawline.
    DIAMOND: The diamond face shape has a narrow chin and forehead, accompanied by wide cheekbones.
    PEAR: The triangular, also known as "pear", face shape has a narrow forehead and larger jawline.
    OBLONG: The oblong face shape is much longer than wide with a very narrow bone structure. (RECTANGLE)
    OVAL: The oval face shape is longer than it is wide with a rounded hairline and narrower jaw than cheekbones.
    """
    
    def __init__(self):
        self.landmarks = []
        self.image = None
        self.face_detector = None
        self.landmark_predictor = None
        self.face_area = None

    def calculate_angle(self, c, b, a):
        """
        Caculates the angle of the jaw using law of cosines 
        :param c: jaw width
        :param b: jaw right-to-down
        :param a: jaw left-to-down
        :return: jaw angle
        """
        # calculate the cosine of the jaw angle 
        cosine_angle = (b**2 + c**2 - a**2) / (2 * b * c)
        jaw_angle_degrees = np.degrees(np.arccos(cosine_angle))
        return jaw_angle_degrees

    def calculate_jawline_slope(self, landmarks):
        """
        Caculates the jawline slope angle
        :param landmarks: landmark points
        :return: slope angle
        """
        start_index = 8
        last_index = 14

        # angles
        result = []
        for i in range(start_index, last_index):
            # set points
            a = landmarks[i - 1]
            b = landmarks[i]
            c = landmarks[i + 1]
            # calculate
            alpha = np.arctan2(a[1] - b[1], b[0] - a[0])
            beta = np.arctan2(b[1] - c[1], c[0] - b[0])
            slope_degrees = np.degrees(beta - alpha)
            result.append(slope_degrees)

        # return
        return result

    def calculate_dist(self, a, b):
        """
        Calculate distance between two points
        :param a: point 1
        :param b: point 2
        :return: distance
        """
        x1, y1 = a
        x2, y2 = b
        distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return distance

    def calculate_jawline(self, landmarks, method="B"):
        """
        Calculate jawline length based on landmarks
        :param landmarks: landmarks data
        :param method: calculation method {"B", "C"}
        :param jawline_start: jawline start point
        :return: jawline length
        """
        # determine default start & last index
        start_index = 8
        last_index = 13
        if method == "C":
            start_index = 5
            last_index = 11

        # set special landmarks
        left_jawline_start = self.mid_point(landmarks[start_index - 1], landmarks[start_index])
        right_jawline_start = self.mid_point(self.landmarks[last_index], self.landmarks[last_index + 1])

        # calculate
        total_distance = 0
        for i in range(start_index, last_index):
            total_distance += self.calculate_dist(landmarks[i], landmarks[i + 1])
        total_distance += self.calculate_dist(landmarks[last_index], right_jawline_start)

        # consider method C
        if method == "C":
            total_distance += self.calculate_dist(landmarks[start_index], left_jawline_start)

        # return jawline
        return total_distance

    def mid_point(self, a, b):
        """
        Get the middle point of two points
        :param a: tuple a
        :param b: tuple b
        :return: mid point in tuple format
        """
        x = np.array(a)
        y = np.array(b)
        mid = (x + y) // 2
        return tuple(mid)

    def top_head_point(self, rect, bottom, mid):
        """
        Get the top head point
        :param rect: face area
        :param bottom: bottom point
        :param mid: mid point
        :return: top head point in tuple
        """
        y = rect.top()
        x = (mid[0] * (bottom[1] - y) - bottom[0] * (mid[1] - y)) // (bottom[1] - mid[1])
        return (x, y)

    def move_point_upward(self, point, dist):
        """
        Move point to upward for distance
        :param point: tuple point
        :param dist: distance value
        """
        return (point[0], point[1] - dist)

    def draw_jawline(self, landmarks, method="B"):
        """
        Draw jawline around the detected face
        :param landmarks: landmark points
        :param method: calculation method {"B", "C"}
        :return: none
        """
        # determine default start & last index
        start_index = 8
        last_index = 13
        if method == "C":
            start_index = 5
            last_index = 11

        # set special landmarks
        left_jawline_start = self.mid_point(landmarks[start_index - 1], landmarks[start_index])
        right_jawline_start = self.mid_point(self.landmarks[last_index], self.landmarks[last_index + 1])

        # draw jawline
        for i in range(start_index, last_index):
            cv2.line(self.image, self.landmarks[i], self.landmarks[i + 1], color=(0, 255, 0), thickness=1)
        cv2.line(self.image, self.landmarks[last_index], right_jawline_start, color=(0, 255, 0), thickness=1)

        # consider method C
        if method == "C":
            cv2.line(self.image, left_jawline_start, self.landmarks[start_index], color=(0, 255, 0), thickness=1)

    def method_a(self, cheek_width, top_jaw_distance, forehead_width, chin_width, head_length, jaw_angle):
        """
        Calculate face shape from 6 parameters
        :param cheek_width: cheek width
        :param top_jaw_distance: top jaw distance
        :param forehead_width: forehead width
        :param chin_width: chin width
        :param head_length: head length
        :param jaw_angle: jaw angle
        :return: face shape in string
        """
        # ratios
        cheek_ratio = cheek_width / head_length
        jaw_ratio = top_jaw_distance / head_length
        forehead_ratio = forehead_width / head_length
        chin_ratio = chin_width / head_length
        head_ratio = head_length / cheek_width

        # initialize return value
        result = "UNKNOWN"

        # round face
        if (
            0.8 <= cheek_ratio <= 1.0 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and
            head_ratio <= 1.25 and jaw_angle <= 50.0
        ):
            result = "ROUND"

        # oval face
        elif (
            0.5 <= cheek_ratio <= 0.8 and
            0.5 <= jaw_ratio <= 0.7 and
            0.5 <= forehead_ratio <= 0.7 and
            0.2 <= chin_ratio <= 0.4 and
            1.25 <= head_ratio <= 1.6 and jaw_angle > 50.0
        ):
            result = "OVAL"

        # oblong face
        elif (
            0.5 <= cheek_ratio <= 0.8 and
            0.5 <= jaw_ratio <= 0.8 and
            0.5 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and
            head_ratio >= 1.30 and jaw_angle > 55
        ):
            result = "OBLONG"

        # square face
        elif (
            0.7 <= cheek_ratio <= 0.99 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.99 and
            0.3 <= chin_ratio <= 0.5 and
            head_ratio <= 1.29 and jaw_angle < 55
        ):
            result = "SQUARE"

        # heart face
        elif (
            0.7 <= cheek_ratio <= 0.8 and
            0.7 <= jaw_ratio <= 0.8 and
            0.5 <= forehead_ratio <= 0.7 and
            0.3 <= chin_ratio <= 0.4 and
            1.2 <= head_ratio <= 1.4
        ):
            result = "HEART"

        # diamond face
        elif (
            0.7 <= cheek_ratio <= 0.8 and
            0.7 <= jaw_ratio <= 0.8 and
            0.6 <= forehead_ratio <= 0.8 and
            0.3 <= chin_ratio <= 0.4 and
            1.2 <= head_ratio <= 1.4
        ):
            result = "DIAMOND"

        # return face shape calculated
        return result

    def method_b(self, forehead_width, cheek_width, face_length, jawline_length):
        """
        Calculate face shape from 4 parameters
        :param forehead_width: forehead width
        :param cheek_width: cheek width
        :param face_length: face length
        :param jawline_length: jawline length
        :return: face shape in string
        """
        # ratios
        cheek_vs_length = cheek_width / face_length
        jawline_vs_cheek = jawline_length / cheek_width
        forehead_vs_cheek = forehead_width / cheek_width
        forehead_vs_jaw = forehead_width / jawline_length
        
        # log
        log(f"Cheek vs Face Length: {cheek_vs_length}", level=2)
        log(f"Jaw vs Cheek: {jawline_vs_cheek}", level=2)
        log(f"Forehead vs Cheek: {forehead_vs_cheek}", level=2)
        log(f"Forehead vs Jaw: {forehead_vs_jaw}", level=2)
        
        # initialize return value
        result = []
        
        # oval face
        if (
            cheek_vs_length <= 0.95 and
            forehead_vs_jaw >= 1.05
        ):
            result.append("OVAL")
        
        # oblong face
        if (
            cheek_vs_length <= 0.95 and
            0.95 <= forehead_vs_jaw <= 1.05
        ):
            result.append("OBLONG")
        
        # round face
        if (
            0.95 <= cheek_vs_length <= 1.0 and
            forehead_vs_cheek <= 0.95 and
            jawline_vs_cheek <= 0.95
        ):
            result.append("ROUND")
        
        # square face
        if (
            0.95 <= cheek_vs_length <= 1.0 and
            forehead_vs_jaw < 0.9
        ):
            result.append("SQUARE")
        
        # pear (triangle) face
        if (
            jawline_vs_cheek >= 1.05 and
            forehead_vs_cheek <= 0.95
        ):
            result.append("TRIANGLE")
        
        # heart face
        if (
            forehead_vs_jaw > 1.05 and
            0.95 <= forehead_vs_cheek <= 1.05
        ):
            result.append("HEART")
        
        # diamond
        if (
            forehead_vs_jaw > 1.05 and
            forehead_vs_cheek < 0.95 and
            cheek_vs_length < 0.95
        ):
            result.append("DIAMOND")
            
        # return
        return result

    def method_c(self, forehead_width, cheek_width, face_length, jawline_length, slope):
        """
        Calculate face shape from 4 parameters
        :param forehead_width: forehead width
        :param cheek_width: cheek width
        :param face_length: face length
        :param jawline_length: jawline length
        :param slope: jawline slope
        :return: face shape in string
        """
        # ratios
        cheek_vs_length = cheek_width / face_length
        jawline_vs_cheek = jawline_length / cheek_width
        forehead_vs_cheek = forehead_width / cheek_width
        forehead_vs_jaw = forehead_width / jawline_length
        chin_angle = slope[0]

        # check if the slope is smooth or hard
        angle_threshold = 21
        jawline_hard = any(element > angle_threshold for element in slope[2:])
        # jaw_slope = slope[0] + slope[1] + slope[2]
        
        # log
        log(f"Cheek vs Face Length: {cheek_vs_length}", level=2)
        log(f"Jaw vs Cheek: {jawline_vs_cheek}", level=2)
        log(f"Forehead vs Cheek: {forehead_vs_cheek}", level=2)
        log(f"Forehead vs Jaw: {forehead_vs_jaw}", level=2)
        log(f"Chin angle: {slope[0]}", level=2)
        log(f"Jaw angle: {slope[2:]}", level=2)
        
        # initialize return value
        result = []
        
        # diamond face
        if (
            cheek_vs_length < 0.8 and
            forehead_vs_cheek < 0.9 and
            forehead_vs_jaw > 1.1
        ):
            result.append("DIAMOND")

        # heart face
        if (
            forehead_vs_cheek > 1.2 and
            forehead_vs_jaw > 1.2
        ):
            result.append("HEART")

        # oblong face
        if (
            cheek_vs_length < 0.79 and
            1.0 <= forehead_vs_cheek <= 1.1 and
            1.0 <= forehead_vs_jaw <= 1.1
        ):
            result.append("OBLONG")

        # oval face
        if (
            0.7 <= cheek_vs_length < 0.79 and
            forehead_vs_cheek < 1.0 and
            forehead_vs_jaw > 1.0 and
            jawline_hard is False
        ):
            result.append("OVAL")

        # round face
        if (
            0.79 <= cheek_vs_length <= 1.1 and
            0.9 <= forehead_vs_jaw <= 1.1 and
            forehead_vs_cheek < 0.95 and
            jawline_vs_cheek < 0.95 and
            jawline_hard is False
        ):
            result.append("ROUND")
        
        # square face
        if (
            0.85 <= cheek_vs_length <= 1.1 and
            0.9 <= forehead_vs_jaw <= 1.1 and
            0.9 <= forehead_vs_cheek <= 1.1 and
            jawline_hard is True
        ):
            result.append("SQUARE")
        
        # pear (triangle) face
        if (
            jawline_vs_cheek > 1.2 and
            forehead_vs_cheek < 0.9
        ):
            result.append("TRIANGLE")
             
        # return
        return result

    def method_d(self, forehead_width, cheek_width, face_length, jaw_slope):
        """
        Calculate face shape from 4 parameters
        :param forehead_width: forehead width
        :param cheek_width: cheek width
        :param face_length: face length
        :param jaw_slope: jawline slope
        :return: face shape in string
        """
        # ratios
        cheek_vs_length = cheek_width / face_length
        forehead_vs_cheek = forehead_width / cheek_width
        slope = jaw_slope[0] + jaw_slope[1] + jaw_slope[2]
        
        # log
        log(f"Cheek vs Face Length: {cheek_vs_length}", level=2)
        log(f"Forehead vs Cheek: {forehead_vs_cheek}", level=2)
        log(f"Jawline slope: {slope}", level=2)
        
        # initialize return value
        result = []

        # oblong face
        if (
            cheek_vs_length <= 0.75
        ):
            result.append("OBLONG")

        # heart face
        if (
            0.75 < cheek_vs_length < 0.9 and
            # jaw_angle < 30 and
            forehead_vs_cheek >= 1.1
        ):
            result.append("HEART")

        # oval face
        if (
            0.75 < cheek_vs_length < 0.9 and
            # jaw_angle < 30 and
            forehead_vs_cheek < 1.1
        ):
            result.append("OVAL")

        # diamond face
        if (
            0.75 < cheek_vs_length < 0.9 and
            # jaw_angle >= 30 and
            forehead_vs_cheek <= 0.9
        ):
            result.append("DIAMOND")

        # pear (triangle) face
        if (
            0.75 < cheek_vs_length < 0.9 and
            # jaw_angle >= 30 and
            1.0 > forehead_vs_cheek > 0.9
        ):
            result.append("TRIANGLE")
        
        # square face
        if (
            cheek_vs_length >= 0.9
            # jaw_angle >= 30
        ):
            result.append("SQUARE")

        # round face
        if (
            cheek_vs_length >= 0.9
            # jaw_angle < 30
        ):
            result.append("ROUND")

        # return
        return result

    def apply_method_a(self, debug=True, save_file=None):
        """
        Get the face shape using method A
        :param debug: flag to display intermediate image
        :param save_file: temporary output file name
        :return: face shape in string
        """
        # set special landmark points for calculation
        cheek_left = self.landmarks[1]
        cheek_right = self.landmarks[15]
        chin_left = self.landmarks[6]
        chin_right = self.landmarks[10]
        jaw_left = self.landmarks[3]
        jaw_right = self.landmarks[13]
        eye_brow_left = self.landmarks[17]
        eye_brow_right = self.landmarks[26]
        bottom_chin = self.landmarks[8]
        
        # for jaw angle calculation
        cheek_bone_right_down_one = self.landmarks[11]

        # calcaulte face landmark distances
        cheek_distance = cheek_right[0] - cheek_left[0]
        top_jaw_distance = jaw_right[0] - jaw_left[0]
        forehead_distance = eye_brow_right[0] - eye_brow_left[0]
        chin_distance = chin_right[0] - chin_left[0]
        head_length = bottom_chin[1] - self.face_area.top()
        
        # jaw angle detection
        jaw_width = top_jaw_distance
        jaw_right_to_down_one = cheek_bone_right_down_one[1] - jaw_right[1]
        jaw_left_to_down_one = cheek_bone_right_down_one[0] - jaw_left[0]
        jaw_angle = self.calculate_angle(jaw_width, jaw_right_to_down_one, jaw_left_to_down_one)

        # determine face shape
        face_shape = self.method_a(cheek_distance, top_jaw_distance, forehead_distance, chin_distance, head_length, jaw_angle)

        if debug:
            # draw face area rectangle
            # cv2.rectangle(self.image, (self.face_area.left(), self.face_area.top()), (self.face_area.right(), self.face_area.bottom()), (0, 255, 0), 1)

            # draw landmarks
            for idx, landmark in enumerate(self.landmarks):
                cv2.putText(self.image, str(idx), landmark, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 255))
                cv2.circle(self.image, landmark, 3, color=(0, 0, 255), thickness=-1)

            # draw cheek width line, face length line, forehead width line
            cv2.line(self.image, cheek_left, cheek_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, chin_left, chin_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, jaw_left, jaw_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, eye_brow_left, eye_brow_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, bottom_chin, (bottom_chin[0], self.face_area.top()), color=(0, 255, 0), thickness=1)

            # save file
            cv2.putText(self.image, str(face_shape), (10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.imwrite(f"./output/{save_file}", self.image)

        # return
        return face_shape
    
    def apply_method_b(self, debug=True, save_file=None):
        """
        Get the face shape using method B
        :param debug: flag to display intermediate image
        :param save_file: temporary output file name
        :return: face shape in string
        """
        # set special landmark points for calculation
        eye_brow_mid = self.mid_point(self.landmarks[19], self.landmarks[24])
        # basic line for measuring forehead width
        temp_line_left = self.mid_point(self.landmarks[0], self.landmarks[17])
        temp_line_right = self.mid_point(self.landmarks[16], self.landmarks[26])
        temp_line_mid = self.mid_point(temp_line_left, temp_line_right)
        # move forehead line
        dist = (eye_brow_mid[1] - self.face_area.top()) // 2 + (temp_line_mid[1] - eye_brow_mid[1])
        forehead_left = self.move_point_upward(temp_line_left, dist)
        forehead_right = self.move_point_upward(temp_line_right, dist)

        cheek_left = self.landmarks[1]
        cheek_right = self.landmarks[15]
        bottom_chin = self.landmarks[8]
        top_head = self.top_head_point(self.face_area, bottom_chin, eye_brow_mid)

        # calculate jawline length
        cheek_width = self.calculate_dist(cheek_left, cheek_right)
        forehead_width = self.calculate_dist(forehead_left, forehead_right)
        face_length = self.calculate_dist(top_head, bottom_chin)
        jawline_length = self.calculate_jawline(self.landmarks)
        log(f"Forehead width: {forehead_width}, Cheek width: {cheek_width}, Face length: {face_length}, Jawline length: {jawline_length}", level=2)

        # determine face shape
        face_shape = self.method_b(forehead_width, cheek_width, face_length, jawline_length)

        if debug:
            # draw face area rectangle
            cv2.rectangle(self.image, (self.face_area.left(), self.face_area.top()), (self.face_area.right(), self.face_area.bottom()), (0, 255, 0), 1)

            # draw landmarks
            for idx, landmark in enumerate(self.landmarks):
                cv2.putText(self.image, str(idx), landmark, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 255))
                cv2.circle(self.image, landmark, 3, color=(0, 0, 255), thickness=-1)

            # draw cheek width line, face length line, forehead width line
            cv2.line(self.image, cheek_left, cheek_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, forehead_left, forehead_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, bottom_chin, top_head, color=(0, 255, 0), thickness=1)
            
            # draw right-down jawline
            self.draw_jawline(self.landmarks, method="B")

            # save file
            cv2.putText(self.image, str(face_shape), (10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.imwrite(f"./output/{save_file}", self.image)

        # return
        return face_shape

    def apply_method_c(self, debug=True, save_file=None):
        """
        Get the face shape using method C
        :param debug: flag to display intermediate image
        :param save_file: temporary output file name
        :return: face shape in string
        """
        # set special landmark points for calculation
        eye_brow_mid = self.mid_point(self.landmarks[19], self.landmarks[24])
        # basic line for measuring forehead width
        temp_line_left = self.mid_point(self.landmarks[0], self.landmarks[17])
        temp_line_right = self.mid_point(self.landmarks[16], self.landmarks[26])
        temp_line_mid = self.mid_point(temp_line_left, temp_line_right)
        # move forehead line
        dist = (eye_brow_mid[1] - self.face_area.top()) // 2 + (temp_line_mid[1] - eye_brow_mid[1])
        forehead_left = self.move_point_upward(temp_line_left, dist)
        forehead_right = self.move_point_upward(temp_line_right, dist)

        cheek_left = self.landmarks[1]
        cheek_right = self.landmarks[15]
        bottom_chin = self.landmarks[8]
        top_head = self.top_head_point(self.face_area, bottom_chin, eye_brow_mid)

        # calculate jawline length
        cheek_width = self.calculate_dist(cheek_left, cheek_right)
        forehead_width = self.calculate_dist(forehead_left, forehead_right)
        face_length = self.calculate_dist(top_head, bottom_chin)
        jawline_length = self.calculate_jawline(self.landmarks, method="C")
        jawline_slope = self.calculate_jawline_slope(self.landmarks)
        log(f"Forehead width: {forehead_width}, Cheek width: {cheek_width}, Face length: {face_length}, Jawline length: {jawline_length}", level=2)

        # determine face shape
        face_shape = self.method_c(forehead_width, cheek_width, face_length, jawline_length, jawline_slope)

        # display intermediate image
        if debug:
            # draw face area rectangle
            # cv2.rectangle(self.image, (self.face_area.left(), self.face_area.top()), (self.face_area.right(), self.face_area.bottom()), (0, 255, 0), 1)

            # draw landmarks
            for idx, landmark in enumerate(self.landmarks):
                cv2.putText(self.image, str(idx), landmark, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 255))
                cv2.circle(self.image, landmark, 3, color=(0, 0, 255), thickness=-1)

            # draw cheek width line, face length line, forehead width line
            cv2.line(self.image, cheek_left, cheek_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, forehead_left, forehead_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, bottom_chin, top_head, color=(0, 255, 0), thickness=1)
            
            # draw jawline (left-bottom-right)
            self.draw_jawline(self.landmarks, method="C")

            # save file
            cv2.putText(self.image, str(face_shape), (10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.imwrite(f"./output/{save_file}", self.image)

        # return
        return face_shape

    def apply_method_d(self, debug=True):
        """
        Get the face shape using method D
        :param debug: flag to display intermediate image
        :return: face shape in string
        """
        # set special landmark points for calculation
        eye_brow_mid = self.mid_point(self.landmarks[19], self.landmarks[24])
        # basic line for measuring forehead width
        temp_line_left = self.mid_point(self.landmarks[0], self.landmarks[17])
        temp_line_right = self.mid_point(self.landmarks[16], self.landmarks[26])
        temp_line_mid = self.mid_point(temp_line_left, temp_line_right)
        # move forehead line
        dist = (eye_brow_mid[1] - self.face_area.top()) // 2 + (temp_line_mid[1] - eye_brow_mid[1])
        forehead_left = self.move_point_upward(temp_line_left, dist)
        forehead_right = self.move_point_upward(temp_line_right, dist)

        cheek_left = self.landmarks[1]
        cheek_right = self.landmarks[15]
        bottom_chin = self.landmarks[8]
        top_head = self.top_head_point(self.face_area, bottom_chin, eye_brow_mid)

        # calculate jawline length
        cheek_width = self.calculate_dist(cheek_left, cheek_right)
        forehead_width = self.calculate_dist(forehead_left, forehead_right)
        face_length = self.calculate_dist(top_head, bottom_chin)
        jawline_slope = self.calculate_jawline_slope(self.landmarks)
        log(f"Forehead width: {forehead_width}, Cheek width: {cheek_width}, Face length: {face_length}", level=2)

        if debug:
            # draw face area rectangle
            # cv2.rectangle(self.image, (self.face_area.left(), self.face_area.top()), (self.face_area.right(), self.face_area.bottom()), (0, 255, 0), 1)

            # draw landmarks
            for idx, landmark in enumerate(self.landmarks):
                cv2.putText(self.image, str(idx), landmark, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0, 255, 255))
                cv2.circle(self.image, landmark, 3, color=(0, 0, 255), thickness=-1)

            # draw cheek width line, face length line, forehead width line
            cv2.line(self.image, cheek_left, cheek_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, forehead_left, forehead_right, color=(0, 255, 0), thickness=1)
            cv2.line(self.image, bottom_chin, top_head, color=(0, 255, 0), thickness=1)
            
            # draw right-down jawline
            for i in range(3, 13):
                cv2.line(self.image, self.landmarks[i], self.landmarks[i + 1], color=(0, 255, 0), thickness=1)

            # save file
            cv2.imwrite("./output/test_d.png", self.image)

        # determine face shape
        face_shape = self.method_d(forehead_width, cheek_width, face_length, jawline_slope)
        return face_shape

    def classify(self, img_path, detector="opencv", method="A", save_file=None):
        """
        Classify face shape into 6 categories {SQUARE, ROUND, PEAR(TRIANGLE), DIAMOND, RECTANGLE, OBLONG}
        :param img_path: input image path
        :param detector: face detector {"opencv", "dlib"}
        :param method: face shape calculation method {"A", "B", ...}
        :param save_file: temporary output file name
        :return: face shape
        """
        # read and resize image
        self.image = cv2.imread(img_path)
        target_width = 800
        height, width, _ = self.image.shape
        ratio = target_width / width
        self.image = cv2.resize(self.image, (target_width, int(height * ratio)))
        log(f"{img_path}", level=0)

        # opencv face and smile detector
        if detector == "opencv":
            face_cascade_path = "./data/haarcascade_frontalface_default.xml"
            if os.path.isfile(face_cascade_path):
                self.face_detector = cv2.CascadeClassifier(face_cascade_path)

        # dlib front face detector
        elif detector == "dlib":
            self.face_detector = dlib.get_frontal_face_detector()

        # detect faces
        faces = None
        
        # opencv detector
        if detector == "opencv":
            # convert the image to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            # apply a Gaussian blur with a 3 x 3 kernel to help remove high frequency noise
            gauss = cv2.GaussianBlur(gray, (3, 3), 0)
            # frame_gray = cv2.equalizeHist(frame_gray)

            # detect faces in the image
            faces = self.face_detector.detectMultiScale(
                gauss,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(100, 100),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

        # dlib detector
        elif detector == "dlib":
            faces = detector(self.image, 0)

        # logging
        log(f"Found {len(faces)} faces!", level=1)

        # landmarks predictor
        predictor_path = f"./data/shape_predictor_68_face_landmarks.dat"
        if os.path.isfile(predictor_path):
            self.landmark_predictor = dlib.shape_predictor(predictor_path)

        for face in faces:
            # opencv face detector
            if detector == "opencv":
                left, top, width, height = face
                bottom = top + height
                right = left + width

                # convertg the opencv rectangle coordinates to Dlib rectangle
                self.face_area = dlib.rectangle(int(left), int(top), int(right), int(bottom))

            # dlib face detector
            elif detector == "dlib":
                self.face_area = face

            # detect landmarks
            shapes = self.landmark_predictor(self.image, self.face_area).parts()
            self.landmarks = [(p.x, p.y) for p in shapes]

            # apply face shape classify methods
            face_shape = "UNDEFINED"
            if method == "A":
                face_shape = self.apply_method_a(debug=True, save_file=save_file)
            elif method == "B":
                face_shape = self.apply_method_b(debug=True)
            elif method == "C":
                face_shape = self.apply_method_c(debug=True, save_file=save_file)
            elif method == "D":
                face_shape = self.apply_method_d(debug=True)

            # return face shape
            log(face_shape, level=2)
            return face_shape


def demo():
    """
    Demo to test classifer
    :param: none
    :return: none
    """
    # image_path = "./test/05.png"
    # image_path = "./faces/round_04.png"
    # image_path = "./faces/oblong_15.png"
    image_path = "./faces/square_10.jpg"
    # image_path = "./faces/oval_10.jpg"
    # image_path = "./output/archived/9LSVC4K.jpg"
    # classifier = FaceShapeClassifier()
    # classifier.classify(image_path, detector='opencv', method='A')

    classifier = FaceShapeClassifier()
    dir_path = "./faces/samples/sample"
    for item in os.listdir(dir_path):
        full_path = f"{dir_path}/{item}"
        classifier.classify(full_path, detector='opencv', method='A', save_file=item)

    # classifier = FaceShapeClassifier()
    # classifier.classify(image_path, detector='opencv', method='B')


if __name__ == "__main__":
    demo()
