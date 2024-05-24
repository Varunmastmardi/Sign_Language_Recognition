import cv2
import numpy as np
import os
import mediapipe as mp
import tkinter as tk
from tkinter import Label
from tkinter import ttk
from PIL import Image, ImageTk
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

# Function from the function.py
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten() if hand_landmarks else np.zeros(21*3)
            return np.concatenate([rh])
    return np.zeros(21*3)

# Loading the trained model
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Setting up MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

actions = np.array(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'])
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

# Tkinter Setup
root = tk.Tk()
root.title("Sign Language Detection")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

# Title label
title_label = ttk.Label(root, text="Sign Language Detection", font=("Helvetica", 24, "bold"), background="#f0f0f0")
title_label.pack(pady=20)

# Frame for video and output
frame = ttk.Frame(root, padding=10)
frame.pack(padx=10, pady=10)

# Label to display the video feed
video_label = Label(frame)
video_label.grid(row=0, column=0, padx=10, pady=10)

# Label to display the detected sign language
output_frame = ttk.LabelFrame(frame, text="Detected Sign", padding=10)
output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

output_label = Label(output_frame, text="", font=("Helvetica", 18), wraplength=200)
output_label.pack(pady=20)

# Label to display the confidence of the detected sign
confidence_label = Label(output_frame, text="", font=("Helvetica", 14), wraplength=200)
confidence_label.pack(pady=20)

# Global variable to control video capture
cap = None

def start_capture():
    global cap
    cap = cv2.VideoCapture(0)
    detect_sign_language()

def stop_capture():
    global cap
    if cap:
        cap.release()
    cap = None

def detect_sign_language():
    global cap, sequence, sentence, accuracy, predictions
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            return
        
        cropframe = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
        image, results = mediapipe_detection(cropframe, mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5))
        
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        try: 
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(f"{res[np.argmax(res)]*100:.2f}")
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(f"{res[np.argmax(res)]*100:.2f}")
                        
                if len(sentence) > 1: 
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]
                
                output_text = f"Sign: {' '.join(sentence)}"
                output_label.config(text=output_text)
                confidence_text = f"Confidence: {''.join(accuracy)}%"
                confidence_label.config(text=confidence_text)
        except Exception as e:
            pass

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, detect_sign_language)
    else:
        video_label.after(10, detect_sign_language)

# Buttons to start and stop the video capture
button_frame = ttk.Frame(root, padding=10)
button_frame.pack(pady=10)

start_button = ttk.Button(button_frame, text="Start Capture", command=start_capture)
start_button.grid(row=0, column=0, padx=10, pady=10)

stop_button = ttk.Button(button_frame, text="Stop Capture", command=stop_capture)
stop_button.grid(row=0, column=1, padx=10, pady=10)

# Start the GUI loop
root.mainloop()
