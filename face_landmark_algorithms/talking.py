from scipy.spatial import distance as dist
import numpy as np


def lips_aspect_ratio(lips):
    """
    Compute the lips aspect ratio based on distances between specific lip landmarks.

    Parameters:
    lips (list): List containing lip landmark coordinates.

    Returns:
    float: Calculated lips aspect ratio.
    """

    # Calculate the vertical distance between the inner top and bottom lips
    dist1 = dist.euclidean(lips[2], lips[6])
    # Calculate the horizontal distance between the corners of the lips
    dist2 = dist.euclidean(lips[0], lips[4])
    # Compute the lips aspect ratio
    distance = float(dist1 / dist2)
    return distance


class Talking:
    # A class variable to store speaking states over frames
    speaking_list = []

    def __init__(self, open_mouth_threshold, duration_speak_frame, open_close_limit):
        """
        Initialize the class with given parameters for talking detection.

        Parameters:
        open_mouth_threshold (float): Lips aspect ratio threshold to determine if the mouth is open.
        duration_speak_frame (int): Number of frames to consider for detecting talking.
        open_close_limit (int): Number of open-close mouth cycles to detect talking.
        """
        self.open_mouth_threshold = open_mouth_threshold
        self.speak_limit = duration_speak_frame
        self.open_close_count = open_close_limit

    def __call__(self, landmark_pts):
        """
        Callable method for the class which allows the object to be used as a function.

        Parameters:
        landmark_pts (list): List containing facial landmark coordinates.

        Returns:
        str: Talking status, either "Talking" or "Not_Talking".
        """
        return self.speaking_condition(landmark_pts)

    def calculate(self, speaking_list):
        """
        Count consecutive groups of 1s (mouth open) and 0s (mouth closed) in the speaking_list.

        Parameters:
        speaking_list (list): List of states indicating if the mouth was open (1) or closed (0).

        Returns:
        tuple: Number of groups of closed mouths and opened mouths.
        """
        # Initialize counters for groups of zeros and ones
        close_mouth = 0
        open_mouth = 0

        # Remember the first value
        previous_value = speaking_list[0]

        # Initialize counter based on the first value
        if previous_value == 0:
            close_mouth += 1
        else:
            open_mouth += 1

        # Iterate through the list starting from the second element
        for value in speaking_list[1:]:
            # Check if the value has changed
            if value != previous_value:
                # Increment the corresponding counter
                if value == 0:
                    close_mouth += 1
                else:
                    open_mouth += 1

                # Update the previous value
                previous_value = value

        return close_mouth, open_mouth

    def speaking_condition(self, face_landmark):
        """
        Determine if someone is talking based on the lips aspect ratio and the defined thresholds.

        Parameters:
        face_landmark (list): List containing facial landmark coordinates.

        Returns:
        str: Talking status, either "Talking" or "Not_Talking".
        """

        # Extract lip landmarks
        lips_points = face_landmark[60:68]
        lear = lips_aspect_ratio(lips_points)

        # Check if the lips aspect ratio exceeds the threshold, indicating an open mouth
        if lear > self.open_mouth_threshold:
            self.speaking_list.append(1)
        else:
            self.speaking_list.append(0)

        # Retain only the recent 'speak_limit' number of entries in the list
        self.speaking_list = self.speaking_list[-self.speak_limit:]

        # Calculate the number of groups of consecutive open and closed mouths
        close_mouth, open_mouth = self.calculate(self.speaking_list)

        # Check if the number of mouth opening cycles matches the threshold
        if open_mouth == self.open_close_count:
            return "Talking"
        else:
            return "Not_Talking"
