import tkinter as tk
import cv2
import pickle
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Labels dictionary for predictions
labels_dict = {0: 'A', 1: 'B', 2: 'c'}

# Initialize mediapipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create a Tkinter window
window = tk.Tk()
window.title("Sign Language Gesture Recognition")
window.geometry("800x600")

# Video capture
cap = cv2.VideoCapture(0)

# Create a canvas to display the video feed
canvas = tk.Canvas(window, width=640, height=480)
canvas.pack()

# Label to display the prediction result
prediction_label = tk.Label(window, text="Prediction: None", font=("Arial", 20))
prediction_label.pack(pady=10)

# Function to update the video feed
def update_frame():
    ret, frame = cap.read()
    if not ret:
        return

    # Process the frame using MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Extract landmarks and predict the gesture
        data_aux = []
        x_ = []
        y_ = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        # Normalize coordinates to the frame size
        H, W, _ = frame.shape
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        # Predict the gesture
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Display the prediction
        prediction_label.config(text=f"Prediction: {predicted_character}")

        # Draw the rectangle around the hand
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Convert the frame to ImageTk format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the canvas with the new frame
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.img_tk = img_tk  # Keep a reference to the image to avoid garbage collection

    # Continue the video feed
    window.after(10, update_frame)

# Start the video feed when the window opens
update_frame()

# Run the Tkinter event loop
window.mainloop()

# Release the camera when the window is closed
cap.release()
cv2.destroyAllWindows()#show me reference images of the output of the code
