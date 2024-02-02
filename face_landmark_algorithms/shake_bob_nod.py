import time
from scipy.spatial import distance as dist
import numpy as np
import cv2


class Head_shaking:
    All_yawn = []  # List to store yaw values over frames.

    def __init__(self, yaw_limit, duration_yaw_frame, alarm_shake_count):
        """
        Initialize the class with given parameters for head shaking detection.

        Parameters:
        yaw_limit (float): Threshold value to determine head movement direction.
        duration_yaw_frame (int): Number of frames to consider for detecting head shaking.
        alarm_shake_count (int): Threshold count to raise an alarm for head shaking.
        """
        self.duration_yaw_frame = duration_yaw_frame
        self.yaw_limit = yaw_limit
        self.alarm_shake_count = alarm_shake_count

    def __call__(self, yaw):
        """Callable method to get head shaking status for a given yaw value."""
        return self.head_shaking(yaw)

    def count_head_shakes(self, yaw_data):
        """Count the number of times head shakes based on yaw data."""
        shake_count = 0
        direction = 0 if yaw_data[0] >= self.yaw_limit else 1  # 0 indicates right, 1 indicates left
        for i in range(1, len(yaw_data)):
            # If direction is right and yaw value goes left beyond the limit
            if direction == 0 and yaw_data[i] < -self.yaw_limit:
                shake_count += 1
                direction = 1
            # If direction is left and yaw value goes right beyond the limit
            elif direction == 1 and yaw_data[i] > self.yaw_limit:
                shake_count += 1
                direction = 0
        return shake_count

    def head_shaking(self, yaw):
        """Determine if the head is shaking based on yaw values."""
        self.All_yawn.append(yaw)
        self.All_yawn = self.All_yawn[-self.duration_yaw_frame:]
        shake_count = self.count_head_shakes(self.All_yawn)
        if shake_count == self.alarm_shake_count:
            return "Head_shaked"
        else:
            return "Head_not_shaked"


class Head_bob:
    All_roll = []  # List to store roll values over frames.

    def __init__(self, duration_rol_frame, roll_limit, alarm_bob_count):
        """Initialize the class with parameters for detecting head bobbing."""
        self.capture_roll_duration = duration_rol_frame
        self.roll_limit = roll_limit
        self.alarm_bob_count = alarm_bob_count

    def __call__(self, roll):
        """Callable method to get head bobbing status for a given roll value."""
        return self.head_bobing(roll)

    def count_head_bob(self, roll_data):
        """Count the number of times head bobs based on roll data."""
        bob_count = 0
        direction2 = 0 if roll_data[0] >= self.roll_limit else 1  # 0 indicates right, 1 indicates left
        for i in range(1, len(roll_data)):
            # Check roll data for bobbing patterns similar to shaking
            if direction2 == 0 and roll_data[i] < -self.roll_limit:
                bob_count += 1
                direction2 = 1
            elif direction2 == 1 and roll_data[i] > self.roll_limit:
                bob_count += 1
                direction2 = 0
        return bob_count

    def head_bobing(self, roll):
        """Determine if the head is bobbing based on roll values."""
        self.All_roll.append(roll)
        self.All_roll = self.All_roll[-self.capture_roll_duration:]
        bobing_count = self.count_head_bob(self.All_roll)
        if bobing_count == self.alarm_bob_count:
            return "Head_Bobing"
        else:
            return "Not_Head_Bobing"


class Head_nod:
    All_pitch = []  # List to store pitch values over frames.

    def __init__(self, pitch_limit, duration_pitch_frame, alarm_nodding_count):
        """Initialize the class with parameters for detecting head nodding."""
        self.duration_pitch_frame = duration_pitch_frame
        self.pitch_limit = pitch_limit
        self.alarm_nodding_count = alarm_nodding_count

    def __call__(self, pitch):
        """Callable method to get head nodding status for a given pitch value."""
        return self.head_nodding(pitch)

    def count_head_nodding(self, pitch_data):
        """Count the number of times head nods based on pitch data."""
        nod_count = 0
        direction = 0 if pitch_data[0] >= self.pitch_limit else 1  # 0 indicates up, 1 indicates down
        for i in range(1, len(pitch_data)):
            # Check pitch data for nodding patterns similar to shaking
            if direction == 0 and pitch_data[i] < -self.pitch_limit:
                nod_count += 1
                direction = 1
            elif direction == 1 and pitch_data[i] > self.pitch_limit:
                nod_count += 1
                direction = 0
        return nod_count

    def head_nodding(self, pitch):
        """Determine if the head is nodding based on pitch values."""
        self.All_pitch.append(pitch)
        self.All_pitch = self.All_pitch[-self.duration_pitch_frame:]
        nodding_count = self.count_head_nodding(self.All_pitch)
        if nodding_count == self.alarm_nodding_count:
            return "Head_Nodding"
        else:
            return "Head_not_Nodding"
