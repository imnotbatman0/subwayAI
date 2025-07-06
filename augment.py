from pynput import keyboard
import pyautogui
import os
import time
import threading
import cv2

folder_mapping = {
    "w": "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/jump",
    "s": "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/roll",
    "a": "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/left",
    "d": "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/right",
    "n" : "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/nope"
}

for key, folder in folder_mapping.items():
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        flipped_img = cv2.flip(img, 1)
        name, ext = os.path.splitext(file)
        new_name = f"{name}f{ext}"
        if(key == "w"):
            cv2.imwrite(os.path.join(folder_mapping["w"], new_name), flipped_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if(key == "s"):
            cv2.imwrite(os.path.join(folder_mapping["s"], new_name), flipped_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if(key == "a"):
            cv2.imwrite(os.path.join(folder_mapping["d"], new_name), flipped_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if(key == "d"):
            cv2.imwrite(os.path.join(folder_mapping["a"], new_name), flipped_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if(key == "n"):
            cv2.imwrite(os.path.join(folder_mapping["n"], new_name), flipped_img, [cv2.IMWRITE_JPEG_QUALITY, 60])

print("all images flipped")