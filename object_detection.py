import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

DEVICE = '0' 
WEIGHTS_PATH = 'models/best.pt'
DATA_PATH = 'data/data.yaml'
SOURCE = 'data/images/orange1.jpg'
IMAGE_SIZE = (480,640)
class ObjectDetection:

    def __init__(self):
        self.device = select_device(DEVICE)
        self.half = self.device.type != 'cpu'
        self.model = DetectMultiBackend(WEIGHTS_PATH, device=self.device, dnn=False, data=DATA_PATH, fp16=self.half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size((480,640), s=stride)  # check image size

        self.bs = 1  # batch_size

        # Run inference
        self.model.warmup(imgsz=(1 if pt else self.bs, 3, *imgsz))  # warmup

    def detect(self,im):
        
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size((480,640), s=stride)
        im0 = im.copy()
       
        
        # Padded resize
        im = letterbox(im, imgsz,stride=stride, auto=32)[0]
        #letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]
        
        # Convert 
        # BGR to RGB, to 3x416x416
        im = im[:, :, ::-1].transpose(2, 0, 1) 
        im = np.ascontiguousarray(im)
        #for path, im, im0s, vid_cap, s in self.dataset:
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
            
        # Inference
        pred = self.model(im, augment=False, visualize=False)

        #NMS
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
        
        
        s=''
        # object bbox_list
        bbox_list=[]

        

        # Process predictions
        for i, det in enumerate(pred):  # per image
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
        
                for *xyxy, conf, cls in reversed(det):
                    #print(xyxy)
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh  
                    #print('before',xywh)  
                    # change scale to pixel
                    xywh[0]=xywh[0]*im0.shape[1]
                    xywh[2]=xywh[2]*im0.shape[1]
                    xywh[1]=xywh[1]*im0.shape[0]
                    xywh[3]=xywh[3]*im0.shape[0]
                    #print('Coordinate ', xywh)
                    temp = []
                    for ts in xywh:
                        temp.append(ts)
                    bbox = list(np.array(temp).astype(int))
                    bbox.append(names[int(cls)])
                    bbox_list.append(bbox)

        #print(s)
        return bbox_list



if __name__ == '__main__':
    with torch.no_grad():
        detector = ObjectDetection()
        im = cv2.imread(SOURCE)
        
        bbox_list = detector.detect(im) 
        print(bbox_list)

        


