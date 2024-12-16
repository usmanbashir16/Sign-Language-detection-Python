import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Define the label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Update based on your classes

# Start the webcam
cap = cv2.VideoCapture(0)  # Try camera index 0 or 1
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Add a short delay to let the camera initialize
time.sleep(2)

# Set camera properties (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height

# Loop to read frames and predict gestures
while True:
    ret, frame = cap.read()

    # Check if the frame was successfully grabbed
    if not ret:
        print("Error: Failed to grab frame. Exiting.")
        break

    # Flip the frame for a more natural view
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    # Convert the frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Prepare data for prediction
    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []
        
        # Process each hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Extract landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            # Normalize the landmarks and prepare the input for prediction
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Print the data being passed to the model for debugging
        print(f"Data for prediction: {data_aux}")

        try:
            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Draw bounding box and predicted character on the frame
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        except Exception as e:
            print(f"Error during prediction: {e}")
            continue  # Continue processing the next frame even if prediction fails

    # Display the frame
    cv2.imshow('Sign Language Detector', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
