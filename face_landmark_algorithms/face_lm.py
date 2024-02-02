import cv2
import numpy as np


class Face_LM:
    def __init__(self, img_size=224):
        """
        Initialize the Face_LM class.

        Parameters:
        img_size (int): Target image size to resize face images.
        """
        self.img_size = img_size
        self.preds_array = np.zeros((68, 2), dtype=int)  # Array to store the 68 face landmarks.
        self.image_size = img_size

    def __call__(self, image, bbox, session, input_name):
        """Callable method to get the face landmarks for a given image and bounding box."""
        return self.face_lm_estimation(image, bbox, session, input_name)

    def face_lm_estimation(self, frame, bbox, session, input_name):
        """
        Estimate face landmarks for a given frame and bounding box.

        Parameters:
        frame (array): Image frame.
        bbox (tuple): Bounding box containing (x1, y1, xm, ym) where x1,y1 is the top-left and xm,ym is the bottom-right corner.
        session: Inference session for the neural network model.
        input_name (str): Input tensor name for the neural network model.

        Returns:
        preds_array (array): Array containing the face landmarks.
        x1 (int): x-coordinate of the top-left corner of the bounding box.
        y1 (int): y-coordinate of the top-left corner of the bounding box.
        """
        x1, xm = int(bbox[0]), int(bbox[2])
        y1, ym = int(bbox[1]), int(bbox[3])

        # Crop the face from the image frame using the bounding box and resize it to the target size.
        image = cv2.resize(frame[y1:ym, x1:xm], (self.image_size, self.image_size))

        #### Preprocessing face image for face landmark estimation
        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)
        image = image.astype(np.float32)
        image /= 127.5  # Normalize the image to [-1, 1]
        image -= 1.0
        image = np.transpose(image, [2, 0, 1])  # Transpose the image dimensions
        image = np.expand_dims(image, axis=0)  # Add a batch dimension
        input_data = image.astype(np.float16)  # Convert to float16 for inference

        # Feed the preprocessed image into the neural network model to get landmarks
        landmark_tuple = session.run(None, {input_name: input_data})

        # Post-process the model outputs to get the final landmark coordinates
        landmark_tuple = landmark_tuple[0][0]
        self.preds_array[:, 0] = landmark_tuple[::2] * w  # x-coordinates of landmarks
        self.preds_array[:, 1] = landmark_tuple[1::2] * h  # y-coordinates of landmarks

        return self.preds_array, x1, y1
