import os
import cv2
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mediapipe as mp

# Prepare the dataset
DATA_DIR = './data'
data = []
labels = []
number_of_classes = 3  # Update this based on your dataset

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Iterate through the dataset
for class_label in range(number_of_classes):
    folder_path = os.path.join(DATA_DIR, str(class_label))
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        frame = cv2.imread(file_path)

        # Process the frame with MediaPipe to get hand landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            data_aux = []
            x_ = []
            y_ = []
            
            # Extract the hand landmarks
            for i in range(len(landmarks.landmark)):
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                
            # Normalize the landmarks
            for i in range(len(landmarks.landmark)):
                x = landmarks.landmark[i].x
                y = landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Append the data and labels
            data.append(data_aux)
            labels.append(class_label)

# Convert data to numpy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'{score * 100}% of samples were classified correctly!')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
