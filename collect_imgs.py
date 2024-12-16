import os
import cv2

# Function to test camera access at a specific index
def test_camera(index):
    cap = cv2.VideoCapture(index)
    if cap.isOpened():
        print(f"Camera found at index {index}")
        cap.release()
        return True
    else:
        print(f"No camera found at index {index}")
        return False

# Try testing camera indices from 0 to 3 (adjust if needed)
camera_index_found = False
for i in range(4):
    if test_camera(i):
        camera_index_found = True
        camera_index = i
        break

if not camera_index_found:
    print("Error: No camera found. Exiting...")
    exit()

# Now that we know a valid camera index, proceed with the data collection

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 26  # For the whole alphabet (A-Z)
dataset_size = 100  # You can reduce this for testing purposes

# Open the camera with the found index
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Error: Could not open camera at index {camera_index}.")
    exit()

# Get the frame width and height for later use in positioning text
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create subfolders for each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    done = False
    while not done:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Failed to capture frame or empty frame received.")
            break

        # Add instruction text on the frame, aligned to the bottom center with a smaller font size
        font_scale = 0.6  # Smaller font size
        font_thickness = 2  # Font thickness
        color = (255, 0, 0)  # Blue color in BGR

        # Split the text into two lines for better visibility
        text1 = f'Class: {chr(65 + j)}.'
        text2 = 'Press "Q" to start collecting or "T" to stop.'

        # Get the text size for both lines
        (text1_width, text1_height), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        (text2_width, text2_height), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Calculate the position for bottom center alignment
        x1 = (W - text1_width) // 2  # Center horizontally for line 1
        x2 = (W - text2_width) // 2  # Center horizontally for line 2
        y1 = H - 40  # 40 pixels from the bottom for line 1
        y2 = H - 10  # 10 pixels from the bottom for line 2

        # Display both lines of text
        cv2.putText(frame, text1, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, text2, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        # Wait for 'Q' key to start collecting data or 'T' to stop
        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            done = True
        elif key == ord('t'):  # Press 'T' to stop collecting and save images
            print("Data collection stopped.")
            cap.release()
            cv2.destroyAllWindows()
            exit()  # Exit the program and close the camera

    # Collect dataset for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            print("Error: Failed to capture frame or empty frame received.")
            break

        # Add class label and image counter on the frame
        cv2.putText(frame, f'Class: {chr(65 + j)}', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Image: {counter + 1}/{dataset_size}', (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('frame', frame)

        # Save the captured image
        cv2.imwrite(os.path.join(DATA_DIR, str(j), f'{counter}.jpg'), frame)

        counter += 1

    print(f'Collected {counter} images for class {j}')

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
