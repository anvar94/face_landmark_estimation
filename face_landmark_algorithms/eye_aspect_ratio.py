from scipy.spatial import distance as dist
import numpy as np


def eye_aspect_rationation(eye):
    """
    Calculate the Eye Aspect Ratio (EAR) for a given eye.

    Parameters:
    eye (array): Coordinates of 6 landmark points of the eye.

    Returns:
    ear (float): Eye Aspect Ratio.
    """
    p2_minus_p6 = dist.euclidean(eye[1], eye[5])
    p3_minus_p5 = dist.euclidean(eye[2], eye[4])
    p1_minus_p4 = dist.euclidean(eye[0], eye[3])
    ear = (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)
    return ear


class Eye_open_close:
    ear_values = []  # List to store the EAR values

    def __init__(self, threshold=0.2):
        """
        Initialize the Eye_open_close class.

        Parameters:
        threshold (float): Threshold value to determine if the eye is open or closed.
        """
        self.threshold = threshold
        self.CALIBRATION_FRAMES = 120  # Number of frames for calibration

    def __call__(self, landmark_pts):
        """Callable method to determine the eye condition for given landmark points."""
        return self.eye_condition(landmark_pts)

    def eye_aspect_ratio(self, pts):
        """
        Calculate the EAR for both left and right eyes.

        Parameters:
        pts (array): Face landmarks.

        Returns:
        aspect_right_eye_pts (float): EAR for the right eye.
        aspect_left_eye_pts (float): EAR for the left eye.
        """
        right_eye_pts = pts[42:48]
        left_eye_pts = pts[36:42]

        aspect_right_eye_pts = eye_aspect_rationation(right_eye_pts)
        aspect_left_eye_pts = eye_aspect_rationation(left_eye_pts)

        return aspect_right_eye_pts, aspect_left_eye_pts

    def eye_condition(self, face_landmark):
        """
        Determine if the eyes are open or closed.

        Parameters:
        face_landmark (array): Face landmarks.

        Returns:
        str: 'EYE_close' if eyes are closed, 'EYE_open' if eyes are open.
        """
        aspect_right_eye_pts, aspect_left_eye_pt = self.eye_aspect_ratio(face_landmark)
        self.ear = (aspect_right_eye_pts + aspect_left_eye_pt) / 2.0

        num_smaller = len([i for i in self.ear_values if i < self.threshold])
        num_bigger = len([i for i in self.ear_values if i >= self.threshold])

        # Append EAR values to the list for calibration purposes
        if self.ear < 0.2 and num_smaller < self.CALIBRATION_FRAMES // 2:
            self.ear_values.append(self.ear)
        elif self.ear > 0.2 and num_bigger < self.CALIBRATION_FRAMES // 2:
            self.ear_values.append(self.ear)

        # Check if enough frames have been calibrated
        if num_smaller > 10 and num_bigger > 10:
            max_closed_eye_value = np.mean(sorted(self.ear_values, reverse=True)[:10])
            min_open_eye_value = min(self.ear_values)
            threshold = (max_closed_eye_value + min_open_eye_value) / 2.0

            if self.ear < threshold:
                return "EYE_close"
            elif self.ear > threshold:
                return "EYE_open"
