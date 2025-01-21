import cv2
import numpy as np
import mediapipe as mp
from tkinter import *
from PIL import Image, ImageTk
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json

# Load model from JSON file
json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Define colors for visualization
colors = [(245,117,16) for _ in range(20)]

# Define detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

# Initialize Tkinter
root = Tk()
root.title("Gesture Recognition App")

# Create a canvas to display video feed
canvas = Canvas(root, width=640, height=480)
canvas.pack()

# Set mediapipe model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Function to update the video feed
    def update_feed():
        global sequence  # Define sequence as global variable
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropframe = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
            image, results = mediapipe_detection(cropframe, hands)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try:
                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
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

            except Exception as e:
                print(e)

            # Display output
            output_text.set("Output: -" + ' '.join(sentence) + ''.join(accuracy))

            # Convert frame to ImageTk format and display on canvas
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            canvas.img = img
            canvas.create_image(0, 0, anchor=NW, image=img)

            # Call this function again after 10ms
            root.after(10, update_feed)

    # Start video capture
    cap = cv2.VideoCapture(0)
    update_feed()

# Run the GUI main loop
root.mainloop()

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
