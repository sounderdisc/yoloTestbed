import sys
sys.path.append("./yolov5/models/")
from models.common import DetectMultiBackend
import torch
import os
from  PIL import Image
import numpy as np


def main():
    # print to make sure we're entering main and in the directory we expect, and have cuda availible
    print("current working irectory: " + os.getcwd() + "\n")
    print("cuda is availible?: " + str(torch.cuda.is_available()))
    # make instance of yolo model, give it weights to load, also set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="bestPizza500.pt", force_reload=True)
    # model = DetectMultiBackend(weights="bestPizza500.pt")
    model.to(device)

    # load an image. in the real thing, this would be a parameter
    im_frame = Image.open("bb8_sample28.png") # true positive, bb8 is in this picture
    # im_frame = Image.open("bb8_paint_true_neg.png") # true negative, bb8 is not in this image
    np_frame = np.array(im_frame)

    # change to be tensor with axes to be (batch, channels, height, width)
    np_frame = np.transpose(np_frame, axes=[2,0,1])
    inputTensor = torch.tensor(np_frame).to(device)
    inputTensor = torch.unsqueeze(inputTensor, 0)
    inputTensor = inputTensor.type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

    # if this isn't ([1, 3, 512, 512]) then youre gonna have an error
    print("image shape: " + str(inputTensor.shape))

    # and this is why i hate weakly typed languages. What is result? WHO KNOWS???
    result = model(np_frame)
    confidence = result.xyxyn[0][:, -2]
    print(confidence)
    if str(confidence) == "tensor([])":
        confidence = -1
    
    print("Did we find BB8?: " + (str(True) if confidence > 0.5 else str(False)))

    # confidence = torch.squeeze(result)
    # confidence = confidence[:,5] # the sixth column should be the confidence scores 
    #max_confidence = torch.max(confidence)

    #print(confidence.shape)
    #print(max_confidence)

    #target_found = True if max_confidence.item() > 0.5 else False

    #print("did we find bb8? :" + str(target_found))

if __name__ == "__main__":
    main()

