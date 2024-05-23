import cv2
import numpy as np
import os
import mediapipe as mp
import tkinter as tk
from tkinter import Label
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

def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

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

# Label to display the video feed
video_label = Label(root)
video_label.pack()

# Label to display the detected sign language
output_label = Label(root, text="", font=("Helvetica", 24))
output_label.pack()

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
                                accuracy.append(str(res[np.argmax(res)] * 100))
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)] * 100))
                        
                if len(sentence) > 1: 
                    sentence = sentence[-1:]
                    accuracy = accuracy[-1:]
                
                output_text = "Output: " + ' '.join(sentence) + ' ' + ''.join(accuracy) + "%"
                output_label.config(text=output_text)
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
start_button = tk.Button(root, text="Start Capture", command=start_capture)
start_button.pack(side=tk.LEFT, padx=10, pady=10)

stop_button = tk.Button(root, text="Stop Capture", command=stop_capture)
stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Start the GUI loop
root.mainloop()
