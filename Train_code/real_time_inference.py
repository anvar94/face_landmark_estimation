import time
import torch.nn as nn
import onnxruntime as rt
import onnxruntime as ort

import torch
import os
from torchvision import transforms
import numpy as np
import cv2
import timm

def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img

class Face_detector:

    def __init__(self, onnx_path="yolox_final.onnx", threshold=0.8):

        self.input_shape = (1, 3, 320, 320)
        self.sess = rt.InferenceSession(onnx_path)
        self.threshold = threshold

    def __call__(self, input):
        return self.infer(input)

    def decode_outputs_np(self, outputs):

        strides = []
        grids = []
        predefined_strides = [8, 16, 32]
        predefined_hw = [[40, 40], [20, 20], [10, 10]]
        for (hsize, wsize), stride in zip(predefined_hw, predefined_strides):
            xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(np.full((shape[0], shape[1], 1), stride))

        grids = np.concatenate(grids, 1)
        strides = np.concatenate(strides, 1)
        outputs = np.concatenate([
            (outputs[..., 0:2] + grids) * strides,
            np.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], -1)
        return outputs

    def infer(self, image):

        # height, width = image.shape[:2]
        input_data = np.zeros(self.input_shape, dtype=np.float32)
        ratio = min(320 / image.shape[0], 320 / image.shape[1])

        input_data[0] = preproc(image, (320, 320))

        input_name = self.sess.get_inputs()[0].name
        output_name = self.sess.get_outputs()[0].name

        # Perform inference using the ONNX model
        outputs = self.sess.run([output_name], {input_name: input_data})
        outputs = self.decode_outputs_np(outputs[0])
        box_corner = np.zeros_like(outputs)

        box_corner[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2
        box_corner[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2
        box_corner[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2] / 2
        box_corner[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3] / 2
        outputs[:, :, :4] = box_corner[:, :, :4]

        output = outputs[0]
        conf = output[:, 4] * output[:, 6]
        conf_mask = conf > 0.5
        conf_masked = conf[conf_mask]

        if not conf_masked.shape[0] == 0:
            conf_masked = conf[conf_mask]
            bboxes = output[conf_mask, :4]
            bboxes /= ratio
            max_index = np.argmax(conf_masked, 0)
            conf_yolox = conf_masked[max_index]
            bbox_yolox = bboxes[max_index]
        else:
            conf_yolox = np.array([])
            bbox_yolox = np.array([])

        return bbox_yolox, conf_yolox
def load_model(model, resume_checkpoints):

    model = nn.DataParallel(model, range(1)).cuda()
    devices = torch.device("cuda:0")
    model.to(devices)

    if os.path.isfile(resume_checkpoints) or os.path.islink(resume_checkpoints):
        pretrained_dict = torch.load(resume_checkpoints)
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']

        model.load_state_dict(pretrained_dict)
    return model


# Load the ONNX model for face detection
face_detector_onnx_path = "/home/hclee/Documents/Face_landmark/face_landmark_algorithms/face_detection/face_detection.onnx"
session = ort.InferenceSession(face_detector_onnx_path, providers=['CUDAExecutionProvider'])

# Retrieve the details of the ONNX session
providers = session.get_providers()
input_name = session.get_inputs()[0].name

# Initialize the face detection model with the ONNX path and detection threshold
face_detector = Face_detector(onnx_path=face_detector_onnx_path, threshold=0.8)

# Set device to CUDA if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the landmark estimation model
checkpoints = "/home/hclee/Documents/Face_landmark/Train_code/checkpoint/ResNet18_224_3.7.pth"
model = timm.create_model("resnet18", num_classes=68 * 2, exportable=True)
model = load_model(model, checkpoints)
model = model.to(device)
model.eval()

# Define image preprocessing steps: Convert to Tensor, then Normalize
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define scaling and image size parameters
scale = 1.12
image_size = 224

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

# Start capturing frames continuously
while (cap.isOpened()):
    # Read a frame from the webcam
    ret, img = cap.read()

    # Check if the frame is successfully captured
    if ret:
        dimg = img.copy()

        # Detect faces in the image
        result_boxes, result_scores = face_detector(img)
        faces = result_boxes

        # If faces are detected, process each face
        if result_boxes.size != 0:
            # Extract the bounding box coordinates of the detected face
            x1, xm = int(result_boxes[0]), int(result_boxes[2])
            y1, ym = int(result_boxes[1]), int(result_boxes[3])
            c1, c2 = (x1, y1), (xm, ym)
            w, h = (xm - x1), (ym - y1)

            # Crop the image to the face and resize it for the landmark model
            crop_img = dimg[y1:ym, x1:xm]
            resize_img = cv2.resize(crop_img, (image_size, image_size))
            resize_img = image_transform(resize_img)
            resize_img = np.expand_dims(resize_img, axis=0)
            imgs = torch.Tensor(resize_img).cuda()

            # Predict the landmarks using the landmark model
            preds_array = model(imgs)
            preds_array = np.asarray(preds_array.to('cpu').detach().numpy(), dtype=np.float32)
            preds_array = preds_array.reshape(-1, 2)
            preds_array = preds_array * (w, h)
            preds_array = np.asarray(preds_array, dtype=int)

            # Draw the landmarks and the bounding box on the image
            for l in range(preds_array.shape[0]):
                color = (0, 0, 255)
                cv2.circle(img, (x1 + preds_array[l][0], y1 + preds_array[l][1]), 1, color, 1)

            color = (0, 0, 255)
            cv2.rectangle(img, c1, c2, color, 2)

            # Display the image with landmarks
            cv2.imshow("image", img)

            # Break out of the loop if 'q' key is pressed
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()