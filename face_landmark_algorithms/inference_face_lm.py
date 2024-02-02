from typing import Any
import onnxruntime as rt
import numpy as np
import cv2
import time
import onnxruntime as ort
from eye_aspect_ratio import Eye_open_close
from yawn_calculate import Lips_open_close
from face_lm import Face_LM
from shake_bob_nod import Head_shaking, Head_bob, Head_nod
from talking import Talking
from head_pose import get_head_pose


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

    # Load the serialized TensorRT engine from file

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


# Load the ONNX model
if __name__ == "__main__":
    # Load the ONNX model for face landmarks estimation
    session = ort.InferenceSession("face_landmark.onnx", providers=['CUDAExecutionProvider'])

    # Check if CUDA execution is supported by the model and set it as provider if it is
    providers = session.get_providers()
    if 'CUDAExecutionProvider' in providers:
        session.set_providers(['CUDAExecutionProvider'])
    input_name = session.get_inputs()[0].name

    # Initialize the default predictions for landmarks
    preds_array = np.zeros((68, 2), dtype=int)
    image_size = 224
    face_detector_onnx_path = "/home/hclee/Documents/Face_landmark/face_landmark_algorithms/face_detection/face_detection.onnx"

    # Start capturing video from the default camera (webcam)
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()

    while webcam.isOpened():
        # Load face detection model
        face_detector = Face_detector(onnx_path=face_detector_onnx_path, threshold=0.8)

        # Read a frame from the webcam
        status, frame = webcam.read()
        image = np.copy(frame)

        # Detect face(s) in the image
        bbox, conf = face_detector(image)

        # If at least one face is detected
        if bbox.size != 0:
            # Process for landmarks estimation
            flm = Face_LM(img_size=224)
            preds_array, x1, y1 = flm(frame, bbox, session, input_name)

            # Check if eyes are open or closed
            statusses = Eye_open_close(threshold=0.2)
            eye_status = statusses(preds_array)

            # Detect if yawning is happening
            lips_status = Lips_open_close(threshold=0.5)
            yawninig = lips_status(preds_array)

            # Detect if talking is happening based on lips movement
            lips_status = Talking(open_mouth_threshold=0.08, duration_speak_frame=40, open_close_limit=3)
            speaking = lips_status(preds_array)

            # Estimate the head pose based on face landmarks
            Pitch, Yaw, Roll = get_head_pose(preds_array, x1, y1, frame)

            # Detect head shaking
            shake_shake = Head_shaking(duration_yaw_frame=40, yaw_limit=25, alarm_shake_count=4)
            head_shake = shake_shake(Yaw)

            # Detect head bobbing movement
            head_bobbing = Head_bob(duration_rol_frame=60, roll_limit=13, alarm_bob_count=3)
            bobbbbing = head_bobbing(Roll)

            # Detect head nodding movement
            head_node = Head_nod(duration_pitch_frame=60, pitch_limit=3, alarm_nodding_count=3)
            nodddding = head_node(Pitch)

            # Create a display of detected activities
            cls = [eye_status, yawninig, speaking, head_shake, bobbbbing, nodddding]
            all_class = [
                'Eye_Status         : ',
                'Yawning_Status     : ',
                'Talking_status     : ',
                'Head_Shake_Status  : ',
                'Head_Bobing_Status : ',
                'Head_Nodding_Status: '
            ]

            # Overlay the detected activities on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_color = (255, 255, 0)
            font_thickness = 2
            for j in range(6):
                text_name = str(all_class[j]) + str(cls[j])
                position = (10, 50 + j * 30)
                cv2.putText(frame, text_name, position, font, font_scale, font_color, font_thickness)

            # Overlay the landmarks and the bounding box on the frame
            for l in range(preds_array.shape[0]):
                cv2.circle(frame, (x1 + preds_array[l][0], y1 + preds_array[l][1]), 1, (0, 0, 255), 1)
            frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])),
                                  (int(bbox[2]), int(bbox[3])), (0, 0, 255), 1)

        # Display the frame
        cv2.imshow("test", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Break if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    webcam.release()
    cv2.destroyAllWindows()


