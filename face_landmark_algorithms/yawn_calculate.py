from scipy.spatial import distance as dist
import numpy as np


def lips_aspect_ratio(lips):
    """
    Compute the lips aspect ratio (LEAR) based on the distance between specific lip landmarks.

    Parameters:
    lips (list): List containing lip landmark coordinates.

    Returns:
    float: Calculated lips aspect ratio.
    """

    # Calculate the vertical distance between the inner top and bottom lips
    dist1 = dist.euclidean(lips[2], lips[6])
    # Calculate the horizontal distance between the corners of the lips
    dist2 = dist.euclidean(lips[0], lips[4])
    # Calculate the lips aspect ratio
    distance = float(dist1 / dist2)
    return distance


class Lips_open_close:
    """
    Class to determine the yawning status based on the lips aspect ratio.
    """

    def __init__(self, threshold=0.5):
        """
        Initialize the class with the given threshold for yawning detection.

        Parameters:
        threshold (float): Lips aspect ratio threshold to differentiate between yawning and not yawning.
                          A higher aspect ratio indicates yawning.
        """
        self.threshold = threshold

    def __call__(self, landmark_pts):
        """
        Callable method for the class which allows the object to be used as a function.

        Parameters:
        landmark_pts (list): List containing all facial landmark coordinates.

        Returns:
        str: Yawning status.
        """
        return self.lips_condition(landmark_pts)

    def lips_condition(self, face_landmark):
        """
        Determine the yawning status based on the lips aspect ratio.

        Parameters:
        face_landmark (list): List containing all facial landmark coordinates.

        Returns:
        str: Yawning status - either "Yawning" or "Not_Yawning".
        """

        # Extract lip landmarks
        lips_points = face_landmark[60:68]
        # Calculate lips aspect ratio
        self.lear = lips_aspect_ratio(lips_points)

        # If the lips aspect ratio is greater than the threshold, it indicates yawning
        if self.lear > self.threshold:
            yawn = "Yawning"
            return yawn

        # If the lips aspect ratio is less than the threshold, it indicates not yawning
        elif self.lear < self.threshold:
            yawn = "Not_Yawning"
            return yawn
