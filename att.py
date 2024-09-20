import cv2
import numpy as np
import os
import datetime
import csv
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
NUM_CLASSES = 10  # Replace with the actual number of classes (people) you have
DATA_FILE = 'face_data.npy'
MODEL_FILE = 'face_recognition_model.h5'
ATTENDANCE_FILE = 'attendance.csv'

# Function to initialize and train the ANN model
def train_model(X, y):
    model = Sequential()
    model.add(Flatten(input_shape=(100, 100)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)
    model.save(MODEL_FILE)

# Function to detect faces and collect data (without using cv2.imshow)
def collect_data():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    face_data = []
    frame_count = 0

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Create a directory to save frames if it doesn't exist
    if not os.path.exists('frames'):
        os.makedirs('frames')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            face_data.append(face_resized)

        if frame_count % 10 == 0 and len(faces) > 0:  # Save less frequently to reduce overhead
            np.save(DATA_FILE, np.array(face_data))

        # Save the current frame instead of displaying it
        cv2.imwrite(f'frames/frame_{frame_count}.jpg', frame)
        frame_count += 1

        # Stop capturing if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to preprocess data and train the model
def preprocess_and_train():
    # Load and preprocess face data
    try:
        face_data = np.load(DATA_FILE)
        labels = np.array([])  # You need to generate or load actual labels for your data

        # Normalize and split data
        X = face_data / 255.0
        y = to_categorical(labels, NUM_CLASSES)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        train_model(X_train, y_train)

    except FileNotFoundError:
        print(f"File '{DATA_FILE}' not found. Make sure to collect data first.")

# Function to perform face recognition and record attendance
def recognize_and_record():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    try:
        model = load_model(MODEL_FILE)
    except FileNotFoundError:
        print(f"Model file '{MODEL_FILE}' not found. Train the model first.")
        return
    
    cap = cv2.VideoCapture(0)
    csv_file = open(ATTENDANCE_FILE, 'a', newline='')
    csv_writer = csv.writer(csv_file)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100))
            face_normalized = face_resized / 255.0
            face_normalized = np.expand_dims(face_normalized, axis=0)

            prediction = model.predict(face_normalized)
            name = np.argmax(prediction)  # Replace with label mapping

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            csv_writer.writerow([name, timestamp])

        # Save the current frame instead of displaying it
        cv2.imwrite(f'frames/recognize_frame_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg', frame)

        # Stop recognizing if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()

# Main execution
if __name__ == "__main__":
    collect_data()  # Collect data (run this only once to gather initial data)
    preprocess_and_train()  # Train the model (after data collection)
    recognize_and_record()  # Recognize faces and record attendance
