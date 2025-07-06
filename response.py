import numpy as np
import os
import PIL.Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pathlib
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from pynput import keyboard
import pyautogui
import time
import threading

time.sleep(2)
model = keras.models.load_model("subai.keras")
image_height = 200
#220
image_width = 200
#344
def image_input(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (image_height, image_width))
    img = img/255.0
    img = np.expand_dims(img, axis=0)
    return img

class_output = ['jump', 'left', 'nope', 'right', 'roll']

input_folder = "/home/thanuj/Semester4/aiml/myown/subwayAI/incoming"


def take_screenshot(folder_path):
    timestamp = int(time.time())
    filename = os.path.join(folder_path, f"{timestamp}.jpg")
    os.system(f"scrot -q 60 '{filename}'")
    return filename

timeout = 0.8

def take_break():
    while(True):
        path = take_screenshot(input_folder)
        img = image_input(path)
        pred_res = model.predict(img)
        pred = np.argmax(pred_res)
        if(pred == 0):
            pyautogui.press('w')
            print("jump with prob: ", pred_res[0][pred])
        elif(pred == 1):
            pyautogui.press('a')
            print("left with prob: ", pred_res[0][pred])
        elif(pred == 3):
            pyautogui.press('d')
            print(" right prob: ", pred_res[0][pred])
        elif(pred == 4):
            pyautogui.press('s')
            print("roll with prob: ", pred_res[0][pred])
        else:
            print("nope with prob: ", pred_res[0][pred])

        os.remove(path)
        time.sleep(timeout)

take_break()