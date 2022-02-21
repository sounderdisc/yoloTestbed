from tkinter import Image
from common import DetectMultiBackend
import os
from  PIL import Image
import numpy as np


def main():
    print("current working irectory: " + os.gwd + "\n")
    model = DetectMultiBackend(weights="bestPizza500.pt")

    im_frame = Image.open("bb8_sample28.png")
    np_frame = np.array(im_frame.getdata())

    # and this is why i hate weakly typed languages. What is result? WHO KNOWS???
    result = model.forward(np_frame)

if __name__ == "__main__":
    main()

