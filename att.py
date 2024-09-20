import os
import cv2
import numpy as np
import datetime
import csv
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
NUM_CLASSES = 10
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
    print(f"Model saved as {MODEL_FILE}")

# Function to detect faces and collect data
def collect_data():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    face_data = []

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

        if len(face_data) > 0:
            np.save(DATA_FILE, np.array(face_data))
            print(f"Saved {len(face_data)} face images to {DATA_FILE}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to preprocess data and train the model
def preprocess_and_train():
    try:
        face_data = np.load(DATA_FILE, allow_pickle=False)
        print(f"Loaded face data of shape: {face_data.shape}")
        labels = np.array([])  # You need to assign labels

        X = face_data / 255.0
        y = to_categorical(labels, NUM_CLASSES)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_model(X_train, y_train)
    except FileNotFoundError:
        print(f"File '{DATA_FILE}' not found. Please collect data first.")
    except ValueError as e:
        print(f"ValueError: {e}. Check if the data is in the correct format.")

# Function to recognize faces and record attendance
def recognize_and_record():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if not os.path.exists(MODEL_FILE):
        print(f"Model file '{MODEL_FILE}' not found. Train the model first.")
        return

    model = load_model(MODEL_FILE)
    cap = cv2.VideoCapture(0)
    csv_file = open(ATTENDANCE_FILE, 'a', newline='')
    csv_writer = csv.writer(csv_file)

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
            name = np.argmax(prediction)

            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            csv_writer.writerow([name, timestamp])

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
