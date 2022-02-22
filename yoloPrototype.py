import sys
sys.path.append("./yolov5/models/")
from common import DetectMultiBackend
import torch
import os
from  PIL import Image
import numpy as np


def main():
    # print to make sure we're entering main and in the directory we expect, and have cuda availible
    print("current working irectory: " + os.getcwd() + "\n")
    print("cuda is availible?: " + str(torch.cuda.is_available()))
    # make instance of yolo model, give it weights to load, also set device
    model = DetectMultiBackend(weights="bestPizza500.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load an image. in the real thing, this would be a parameter
    im_frame = Image.open("bb8_sample28.png")
    np_frame = np.array(im_frame)

    # change to be tensor with axes to be (batch, channels, height, width)
    np_frame = np.transpose(np_frame, axes=[2,0,1])
    inputTensor = torch.tensor(np_frame).to(device)
    inputTensor = torch.unsqueeze(inputTensor, 0)

    print("image shape: " + str(inputTensor.shape))

    # and this is why i hate weakly typed languages. What is result? WHO KNOWS???
    result = model.forward(inputTensor)

if __name__ == "__main__":
    main()

