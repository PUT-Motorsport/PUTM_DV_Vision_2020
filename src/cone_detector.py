#!/usr/bin/env python3

from __future__ import print_function

import ctypes

import os
import cv2
import yaml 
import time
import numpy as np
import rospy as rp
# import tensorrt as trt
# import pycuda.autoinit
# import pycuda.driver as cuda


# try:
#     ctypes.cdll.LoadLibrary('/home/putm/dv_ws/src/putm_dv_vision/src/plugins/libyolo_layer.so')
# except OSError as e:
#     raise SystemExit('ERROR: failed to load /home/putm/dv_ws/src/putm_dv_vision/src/plugins/libyolo_layer.so.  '
#                      'Did you forget to do a "make" in the "/home/putm/dv_ws/src/putm_dv_vision/src/plugins/" '
#                      'subdirectory?') from e


class ConeDetectorOpenCV:
    def __init__(self):
        self.yolo_weights_file = rp.get_param('/models/yolo/opencv/yolo_weights_file')
        self.yolo_config_file = rp.get_param('/models/yolo/opencv/yolo_config_file')
        self.confidence_threshold = rp.get_param('/models/yolo/opencv/confidence_threshold')
        self.nms_threshold = rp.get_param('/models/yolo/opencv/nms_threshold')
        self.model_size = rp.get_param('/models/yolo/opencv/model_size')

        self.net = cv2.dnn.readNetFromDarknet(self.yolo_config_file, self.yolo_weights_file)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(self.model_size, self.model_size), scale=1/255)

        if self.net is None: 
            print("Error loading cone detector model")


    def predict(self, img: np.ndarray) -> np.ndarray:
        """Detects cones in image.

        Parameters
        ----------
        img : np.ndarray
            Image data array.

        Returns
        -------
        np.ndarray
            Array which contain detection boxes in form [x,y,w,h].
        """
        _, _, boxes = self.model.detect(img, confThreshold=self.confidence_threshold, nmsThreshold=self.nms_threshold)

        return np.array(boxes)


# class HostDeviceMem(object):
#     """Simple helper data class that's a little nicer to use than a 2-tuple."""
#     def __init__(self, host_mem, device_mem):
#         self.host = host_mem
#         self.device = device_mem

#     def __str__(self):
#         return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

#     def __repr__(self):
#         return self.__str__()


# class ConeDetectorTRT:
#     """TrtYOLO class encapsulates things needed to run TRT YOLO."""
#     def _load_engine(self):
#         with open(self.model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
#             return runtime.deserialize_cuda_engine(f.read())


#     def _get_input_shape(self, engine):
#         """Get input shape of the TensorRT YOLO engine."""
#         binding = engine[0]
#         assert engine.binding_is_input(binding)
#         binding_dims = engine.get_binding_shape(binding)
#         if len(binding_dims) == 4:
#             return tuple(binding_dims[2:])
#         elif len(binding_dims) == 3:
#             return tuple(binding_dims[1:])
#         else:
#             raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))


#     def _allocate_buffers(self, engine):
#         """Allocates all host/device in/out buffers required for an engine."""
#         inputs = []
#         outputs = []
#         bindings = []
#         output_idx = 0
#         stream = cuda.Stream()
#         assert 3 <= len(engine) <= 4  # expect 1 input, plus 2 or 3 outpus
#         for binding in engine:
#             binding_dims = engine.get_binding_shape(binding)
#             if len(binding_dims) == 4:
#                 # explicit batch case (TensorRT 7+)
#                 size = trt.volume(binding_dims)
#             elif len(binding_dims) == 3:
#                 # implicit batch case (TensorRT 6 or older)
#                 size = trt.volume(binding_dims) * engine.max_batch_size
#             else:
#                 raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))
#             dtype = trt.nptype(engine.get_binding_dtype(binding))
#             # Allocate host and device buffers
#             host_mem = cuda.pagelocked_empty(size, dtype)
#             device_mem = cuda.mem_alloc(host_mem.nbytes)
#             # Append the device buffer to device bindings.
#             bindings.append(int(device_mem))
#             # Append to the appropriate list.
#             if engine.binding_is_input(binding):
#                 inputs.append(HostDeviceMem(host_mem, device_mem))
#             else:
#                 # each grid has 3 anchors, each anchor generates a detection
#                 # output of 7 float32 values
#                 assert size % 7 == 0
#                 outputs.append(HostDeviceMem(host_mem, device_mem))
#                 output_idx += 1
#         return inputs, outputs, bindings, stream


#     def __init__(self, model_path):
#         """Initialize TensorRT plugins, engine and conetxt."""
#         self.model_path = rp.get_param('/models/yolo/opencv/yolo_weights_file')
#         self.confidence_threshold = rp.get_param('/models/yolo/opencv/confidence_threshold')
#         self.nms_threshold = rp.get_param('/models/yolo/opencv/nms_threshold')

#         self.trt_logger = trt.Logger(trt.Logger.INFO)
#         self.engine = self._load_engine()

#         self.input_shape = self._get_input_shape(self.engine)

#         try:
#             self.context = self.engine.create_execution_context()
#             self.inputs, self.outputs, self.bindings, self.stream = \
#                 self.allocate_buffers(self.engine)
#         except Exception as e:
#             raise RuntimeError('fail to allocate CUDA resources') from e


#     def __del__(self):
#         """Free CUDA memories."""
#         del self.outputs
#         del self.inputs
#         del self.stream


#     def do_inference(self, context, bindings, inputs, outputs, stream):
#         """do_inference (for TensorRT 7.0+)
#         This function is generalized for multiple inputs/outputs for full
#         dimension networks.
#         Inputs and outputs are expected to be lists of HostDeviceMem objects.
#         """
#         # Transfer input data to the GPU.
#         [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
#         # Run inference.
#         context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
#         # Transfer predictions back from the GPU.
#         [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
#         # Synchronize the stream
#         stream.synchronize()
#         # Return only the host outputs.
#         return [out.host for out in outputs]


#     def _preprocess_yolo(self, img, input_shape):
#         """Preprocess an image before TRT YOLO inferencing.
#         # Args
#             img: int8 numpy array of shape (img_h, img_w, 3)
#             input_shape: a tuple of (H, W)
#         # Returns
#             preprocessed img: float32 numpy array of shape (3, H, W)
#         """
#         img = cv2.resize(img, (input_shape[1], input_shape[0]))

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.transpose((2, 0, 1)).astype(np.float32)
#         img /= 255.0
#         return img


#     def _nms_boxes(detections, nms_threshold):
#         """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
#         boxes with their confidence scores and return an array with the
#         indexes of the bounding boxes we want to keep.
#         # Args
#             detections: Nx7 numpy arrays of
#                         [[x, y, w, h, box_confidence, class_id, class_prob],
#                         ......]
#         """
#         x_coord = detections[:, 0]
#         y_coord = detections[:, 1]
#         width = detections[:, 2]
#         height = detections[:, 3]
#         box_confidences = detections[:, 4] * detections[:, 6]

#         areas = width * height
#         ordered = box_confidences.argsort()[::-1]

#         keep = list()
#         while ordered.size > 0:
#             # Index of the current element:
#             i = ordered[0]
#             keep.append(i)
#             xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
#             yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
#             xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
#             yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

#             width1 = np.maximum(0.0, xx2 - xx1 + 1)
#             height1 = np.maximum(0.0, yy2 - yy1 + 1)
#             intersection = width1 * height1
#             union = (areas[i] + areas[ordered[1:]] - intersection)
#             iou = intersection / union
#             indexes = np.where(iou <= nms_threshold)[0]
#             ordered = ordered[indexes + 1]

#         keep = np.array(keep)
#         return keep


#     def _postprocess_yolo(self, trt_outputs, img_w, img_h, conf_th, nms_threshold, input_shape):
#         """Postprocess TensorRT outputs.
#         # Args
#             trt_outputs: a list of 2 or 3 tensors, where each tensor
#                         contains a multiple of 7 float32 numbers in
#                         the order of [x, y, w, h, box_confidence, class_id, class_prob]
#             conf_th: confidence threshold
#         # Returns
#             boxes, scores, classes (after NMS)
#         """
#         # filter low-conf detections and concatenate results of all yolo layers
#         detections = []
#         for o in trt_outputs:
#             dets = o.reshape((-1, 7))
#             dets = dets[dets[:, 4] * dets[:, 6] >= conf_th]
#             detections.append(dets)
#         detections = np.concatenate(detections, axis=0)

#         if len(detections) == 0:
#             boxes = np.zeros((0, 4), dtype=np.int)
#             scores = np.zeros((0,), dtype=np.float32)
#             classes = np.zeros((0,), dtype=np.float32)
#         else:
#             box_scores = detections[:, 4] * detections[:, 6]

#             # scale x, y, w, h from [0, 1] to pixel values
#             old_h, old_w = img_h, img_w
#             offset_h, offset_w = 0, 0

#             detections[:, 0:4] *= np.array(
#                 [old_w, old_h, old_w, old_h], dtype=np.float32)

#             # NMS
#             nms_detections = np.zeros((0, 7), dtype=detections.dtype)
#             for class_id in set(detections[:, 5]):
#                 idxs = np.where(detections[:, 5] == class_id)
#                 cls_detections = detections[idxs]
#                 keep = self._nms_boxes(cls_detections, nms_threshold)
#                 nms_detections = np.concatenate(
#                     [nms_detections, cls_detections[keep]], axis=0)

#             xx = nms_detections[:, 0].reshape(-1, 1)
#             yy = nms_detections[:, 1].reshape(-1, 1)
#             ww = nms_detections[:, 2].reshape(-1, 1)
#             hh = nms_detections[:, 3].reshape(-1, 1)
#             boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
#             boxes = boxes.astype(np.int)
#             scores = nms_detections[:, 4] * nms_detections[:, 6]
#             classes = nms_detections[:, 5]
#         return boxes, scores, classes


#     def detect(self, img):
#         """Detect objects in the input image."""
#         img_resized = self._preprocess_yolo(img, self.input_shape)

#         # Set host input to the image. The do_inference() function
#         # will copy the input to the GPU before executing.
#         self.inputs[0].host = np.ascontiguousarray(img_resized)
#         trt_outputs = self.do_inference(
#             context=self.context,
#             bindings=self.bindings,
#             inputs=self.inputs,
#             outputs=self.outputs,
#             stream=self.stream)

#         boxes, scores, classes = self._postprocess_yolo(
#             trt_outputs, img.shape[1], img.shape[0], self.confidence_threshold,
#             nms_threshold=self.nms_threshold, input_shape=self.input_shape)

#         # clip x1, y1, x2, y2 within original image
#         boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img.shape[1]-1)
#         boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img.shape[0]-1)

#         # format bounding boxes from ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
#         xy_min = np.hstack([np.zeros((boxes.shape[0], 2)), boxes[:,:2]])
#         bboxes = np.subtract(boxes, xy_min)

#         return bboxes, scores, classes
