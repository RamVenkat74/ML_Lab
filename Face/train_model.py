import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

dataset_path = "dataset"

faces = []
labels = []
names = []
label_id = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    # Skip if it's not a directory (e.g., random files in the dataset folder)
    if not os.path.isdir(person_path):
        continue

    # Add the folder name (person's name) to the list
    names.append(person)

    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path, image_name)

        # Read the image in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # SAFETY CHECK: Only process the image if it successfully loaded
        if img is not None:
            img = cv2.resize(img, (100, 100))  # resize to 100x100
            faces.append(img.flatten())
            labels.append(label_id)
        else:
            print(f"Warning: Could not read image at {img_path}. Skipping.")

    # Increment the label ID for the next person
    label_id += 1

# Convert lists to numpy arrays and normalize pixel values to be between 0 and 1
faces = np.array(faces) / 255.0
labels = np.array(labels)

print("Training model...")

# Initialize and train the Multi-Layer Perceptron (Neural Network)
model = MLPClassifier(hidden_layer_sizes=(
    256, 128), max_iter=500, verbose=True)
model.fit(faces, labels)

# Save the trained model and the name mappings to disk
pickle.dump(model, open("face_model.pkl", "wb"))
pickle.dump(names, open("names.pkl", "wb"))

print("✅ Model trained and saved successfully!")
