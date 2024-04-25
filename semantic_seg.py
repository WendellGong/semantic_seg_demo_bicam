import torch
import cv2
import numpy as np

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)

if __name__ == "__main__":
    main()