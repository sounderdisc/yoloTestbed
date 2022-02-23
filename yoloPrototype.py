import sys
sys.path.append("./yolov5/models/")
from common import DetectMultiBackend
import torch
import cv2 as cv
import os
from  PIL import Image
import numpy as np


def main():
    # print to make sure we're entering main and in the directory we expect, and have cuda availible
    print("current working irectory: " + os.getcwd() + "\n")
    print("cuda is availible?: " + str(torch.cuda.is_available()))
    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.to(device)

    # load an image. in the real thing, this would be a parameter
    # im_frame = Image.open("bb8_sample28.png") # true positive, bb8 is in this picture
    im_frame = Image.open("bb8_paint_true_neg.png") # true negative, bb8 is not in this image

    # all we have to do is wrap the image into a batch. I guess torch hub deals with reordering data for us?
    imgs = [im_frame]
    
    # Inference
    results = model(imgs, size=512)  # includes NMS

    # show shapes ad results
    # print(imgs.shape)
    results.print()

    confidence = results[:, :, 5] # the sixth column should be the confidence scores 
    max_confidence = confidence.max()

    print(max_confidence)

    target_found = True if max_confidence.item() > 0.5 else False

    print("did we find bb8? :" + str(target_found))

if __name__ == "__main__":
    main()

