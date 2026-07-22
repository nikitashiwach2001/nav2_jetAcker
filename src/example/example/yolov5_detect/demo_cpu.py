
import onnxruntime
import numpy as np
import ctypes
import time
import cv2
import random
import torch


LEN_ALL_RESULT = 38001
LEN_ONE_RESULT = 38

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()  # create instance for 'from utils.plots import colors'

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


class YoLov5ONNX(object):
    """
    description: A YOLOv5 class that wraps ONNX model, preprocess and postprocess ops.
    """

    def __init__(self, model_path, classes, conf_thresh=0.1, iou_threshold=0.1):
        self.CONF_THRESH = conf_thresh
        self.IOU_THRESHOLD = iou_threshold
        
        # load labels
        self.categories = classes

        # Load the ONNX model
        self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        # Get input and output details
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_w = self.input_shape[3]
        self.input_h = self.input_shape[2]

        model_outputs = self.session.get_outputs()
        self.output_name = model_outputs[0].name

    def infer(self, raw_image_generator):
        # Convert list to iterator if it is not already
        if isinstance(raw_image_generator, list):
            raw_image_generator = iter(raw_image_generator)

        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[1, 3, self.input_h, self.input_w], dtype=np.float32)
        
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(next(raw_image_generator))
        batch_image_raw.append(image_raw)
        batch_origin_h.append(origin_h)
        batch_origin_w.append(origin_w)
        np.copyto(batch_input_image[0], input_image)
        
        batch_input_image = np.ascontiguousarray(batch_input_image, dtype=np.float32)

        # Run inference
        start = time.time()
        output = self.session.run([self.output_name], {self.input_name: batch_input_image})[0]
        end = time.time()

        # Do postprocess
        boxes, scores, classid = self.post_process(output, batch_origin_h[0], batch_origin_w[0])
        return boxes, scores, classid 

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate width and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy array of shape [1, num_boxes, 85] (for COCO dataset)
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score corresponding to box
            result_classid: finally classid, a numpy, each element is the classid corresponding to box
        """
        # Remove batch dimension
        output = output[0]

        # Filter out object confidence scores below threshold
        conf_mask = output[:, 4] >= self.CONF_THRESH
        output = output[conf_mask]

        # Rescale boxes from [0, 1] to original image dimensions
        boxes = output[:, :4]
        boxes[:, 0] *= origin_w
        boxes[:, 2] *= origin_w
        boxes[:, 1] *= origin_h
        boxes[:, 3] *= origin_h

        # Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2]
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1] -= boxes[:, 3] / 2
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        # Extract class scores and class ids
        scores = output[:, 4]
        classid = output[:, 5:].argmax(axis=1)

        # Perform non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.CONF_THRESH, self.IOU_THRESHOLD)
        if len(indices) > 0:
            indices = indices.flatten()
            result_boxes = boxes[indices]
            result_scores = scores[indices]
            result_classid = classid[indices]
        else:
            result_boxes = np.array([])
            result_scores = np.array([])
            result_classid = np.array([])

        return result_boxes, result_scores, result_classid

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(1):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)
if __name__ == "__main__":
    import os
    classes = ['go', 'right', 'park', 'red', 'green', 'crosswalk']
    yolov5_wrapper = YoLov5ONNX('./best.onnx', classes)
    frame = cv2.imread('/home/ubuntu/third_party_ros2/my_data/JPEGImages/image_1.jpg')
    boxes, scores, classid = yolov5_wrapper.infer([frame])
    for box, cls_conf, cls_id in zip(boxes, scores, classid):
        color = colors(cls_id, True)
        plot_one_box(
        box,
        frame,
        color=color,
        label="{}:{:.2f}".format(
            classes[cls_id], cls_conf
        ),
    )
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
