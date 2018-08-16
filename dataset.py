import os
import cv2
import numpy as np
from scipy import misc
from model import *
import torch

def get_real_image(image_size=64, input_path="", test_size=0): # path 불러오기
    images = []

    file_list = os.listdir(input_path)

    for file in file_list:
        if file.endswith(".jpg") or file.endswith(".png"):
            file_path = os.path.join(input_path, file)

            image = cv2.imread(file_path)  # fn이 한글 경로가 포함되어 있으면 제대로 읽지 못함. binary로 바꿔서 처리하는 방법있음

            if image is None:
                print("None")
                continue
            # image를 image_size(default=64)로 변환
            image = cv2.resize(image, (image_size, image_size))
            image = image.astype(np.float32) / 255.
            image = image.transpose(2, 0, 1)
            images.append(image)

    if images:
        print("push image in the stack")
        images = np.stack(images)
    else:
        print("error, images is emtpy")
    return images[:test_size], images[test_size:]





def save_all_image(save_path, generator, A, A_gt):

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    print(A[0][1].shape)
    g_image = generator(A)

    A = A.cpu().data.numpy()

    for i in range(0, len(A)):
        for j in range(0,12 ,3):
            save_image(str(i) + "_" + str(j), np.array([A[i][j], A[i][j+1], A[i][j+2]]), save_path) #RGB
        save_image(str(i) + "_T", A_gt[i], save_path)
        save_image(str(i) + "_F", g_image[i].cpu().data.numpy(), save_path)

    print("complete to save image")


def save_image(name, image, result_path):
    image = image.transpose(1, 2, 0) * 255.
    misc.imsave(os.path.join(result_path, name + '.jpg'), image.astype(np.uint8)[:, :, ::-1])
