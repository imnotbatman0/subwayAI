from pynput import keyboard
import pyautogui
import os
import time
import threading

time.sleep(5)
folder_mapping = {
    "w": "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/jump",
    "s": "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/roll",
    "a": "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/left",
    "d": "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/right",
}

default_folder = "/home/thanuj/Semester4/aiml/myown/subwayAI/dataset/nope"

last_key_time = time.time()
timer = None
timeout = 0.25  # seconds

import os
import time
from PIL import Image

def take_screenshot(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    timestamp = int(time.time())
    
    original_path = os.path.join(folder_path, f"{timestamp}.jpg")
    flipped_path = os.path.join(folder_path, f"{timestamp}_flipped.jpg")

    os.system(f"scrot -q 60 '{original_path}'") 

    if os.path.exists(original_path):
        image = Image.open(original_path)
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)  
        flipped_image.save(flipped_path, quality=60)  

    print(f"original saved: {original_path}")
    print(f"flipped  saved: {flipped_path}")




def check_inactivity():
    """Takes a screenshot if no key is pressed for timeout seconds."""
    global last_key_time, timer
    while True:
        time.sleep(1) 
        if time.time() - last_key_time >= timeout:
            take_screenshot(default_folder)
            last_key_time = time.time()  

def on_press(key):
    """Handles key press events and resets the inactivity timer."""
    global last_key_time
    last_key_time = time.time() 
    
    key_char = key.char if hasattr(key, 'char') else None
    if key_char in folder_mapping:
        take_screenshot(folder_mapping[key_char])

inactivity_thread = threading.Thread(target=check_inactivity, daemon=True)
inactivity_thread.start()

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
