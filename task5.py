import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained ArcFace model
model = load_model('path_to_arcface_model.h5')

# Function to detect faces in an image
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

# Function to preprocess and embed face images
def preprocess_image(image):
    # Preprocess image (resize, normalize, etc.)
    processed_image = cv2.resize(image, (112, 112))
    processed_image = np.expand_dims(processed_image, axis=0)
    processed_image = processed_image / 255.0  # Normalize to [0, 1]
    return processed_image

# Function to extract face embeddings using ArcFace model
def extract_face_embeddings(image):
    processed_image = preprocess_image(image)
    embeddings = model.predict(processed_image)
    return embeddings

# Load an image
image = cv2.imread('path_to_image.jpg')

# Detect faces
faces = detect_faces(image)

# Example database of known face embeddings (for demonstration)
# In practice, load or compute these embeddings and labels
known_face_embeddings = [np.random.rand(1, 512) for _ in range(5)]
known_face_labels = ['Person 1', 'Person 2', 'Person 3', 'Person 4', 'Person 5']

# Draw rectangles around the detected faces and recognize them
for (x, y, w, h) in faces:
    face_image = image[y:y+h, x:x+w]  # Extract the face ROI
    embeddings = extract_face_embeddings(face_image)
    
    # Recognize face by comparing embeddings with the database
    min_dist = float('inf')
    label = 'Unknown'
    for i, known_embedding in enumerate(known_face_embeddings):
        dist = np.linalg.norm(embeddings - known_embedding)
        if dist < min_dist:
            min_dist = dist
            label = known_face_labels[i]
    
    # Draw the rectangle and label on the image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Faces and Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
