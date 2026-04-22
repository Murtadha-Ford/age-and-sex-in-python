import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('model_prediction_1.h5')

# Function to predict age and sex from an image frame
def predict_age_sex(frame, model):
    img = cv2.resize(frame, (48, 48))
    img = np.reshape(img, [48, 48, 3])
    img = img / 255.0  # Normalize the image

    pred_1 = model.predict(np.array([img]))
    sex_f = ['Male', 'Female']
    age = int(np.round(pred_1[1][0]))
    sex = int(np.round(pred_1[0][0]))

    return age, sex_f[sex]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict age and sex
    age, sex = predict_age_sex(frame, model)

    # Display the predictions on the frame
    cv2.putText(frame, f'Age: {age-10}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Sex: {sex}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Age and Sex Prediction', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
