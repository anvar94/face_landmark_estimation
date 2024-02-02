# Import necessary modules
from yolox import YOLOX_TRT  # Import the TensorRT implementation of YOLOX for object detection
from load_engine_file import *  # Custom module to load the TensorRT engine file
from torchvision import transforms
import numpy as np
import cv2

# Initialize object detector using YOLOX
obj_detector = YOLOX_TRT("yolox_m_icms_4cls.engine")
print(f"loading object odetector from:  done.")
obj_detector.destroy()

# Load the inference engine for landmark prediction
engine_path = "shufflenetv2_output.engine"
engine = get_engine(engine_path)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# Define some constants
image_size = 256

# Start the video capture using the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Initialize an array to hold landmark predictions
preds_array = np.zeros((68, 2), dtype=int)
color = (0, 0, 255)
color_lm = (50, 205, 50)

# Process video frames in a loop
while (cap.isOpened()):
    ret, img = cap.read()
    if ret == True:
        start = time.time()

    dimg = img.copy()

    # Perform object detection on the current frame
    result_boxes, result_scores, result_classid = obj_detector.infer(img)

    # If a bounding box was detected, process it
    if len(result_boxes) > 0:
        start_time = time.time()

        # Extract the bounding box coordinates
        x1, xm = int(result_boxes[0][0]), int(result_boxes[0][2])
        y1, ym = int(result_boxes[0][1]), int(result_boxes[0][3])
        c1, c2 = (x1, y1), (xm, ym)
        w, h = (xm - x1), (ym - y1)

        # Resize and preprocess the region inside the bounding box
        image = cv2.resize(dimg[y1:ym, x1:xm], (image_size, image_size))
        if (len(image.shape) == 2):
            image = image.reshape(image.shape[0], image.shape[1], 1)
        image = image.astype(np.float32)
        image /= 127.5
        image -= 1.0
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        im = np.ascontiguousarray(image)

        # Infer landmarks on the preprocessed image
        inputs[0].host = im.reshape(-1)
        landmark_tuple = Do_Inference(context, bindings, inputs, outputs, stream)[0]

        # Convert the landmarks to the original image scale
        preds_array[:, 0] = landmark_tuple[::2] * w
        preds_array[:, 1] = landmark_tuple[1::2] * h

        # Calculate and display FPS
        FPS = int(1. / (time.time() - start_time))
        print("FPS: ", FPS)

        # Draw the bounding box and landmarks on the original image
        cv2.rectangle(img, c1, c2, color, 2)
        for l in range(preds_array.shape[0]):
            cv2.circle(img, (x1 + preds_array[l][0], y1 + preds_array[l][1]), 1, color_lm, 1)

    # Display the annotated image
    cv2.imshow("image", img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
