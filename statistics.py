# import numpy as np
# import pandas as pd
# import cv2
# import os
# import mediapipe as mp
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report


# mp_face_mesh = mp.solutions.face_mesh


# num_images = 1470  # Replace with the actual number of images
# labels = np.random.randint(0, 2, num_images)  # Random labels for binary classification


# def calculate_distance(point1, point2):
#     return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# def extract_features(landmarks):
#     features = []
#     for points in landmarks:
#         feature_vector = []
#         # Example: Outer mouth distance calculation (points 48 to 59)
#         feature_vector.append(calculate_distance(points[48], points[59]))
#         # Add other feature calculations as per Tables 1 and 2
#         feature_vector.append(calculate_distance(points[60], points[67])) # Inner mouth
#         feature_vector.append(calculate_distance(points[17], points[21])) # Right eyebrow
#         feature_vector.append(calculate_distance(points[22], points[26])) # Left eyebrow
#         feature_vector.append(calculate_distance(points[36], points[41])) # Right eye
#         feature_vector.append(calculate_distance(points[42], points[47])) # Left eye
#         feature_vector.append(calculate_distance(points[27], points[34])) # Nose
#         feature_vector.append(calculate_distance(points[0], points[16]))  # Jaw
#         # Add width calculations
#         feature_vector.append(calculate_distance(points[0], points[16]))  # Forehead width
#         feature_vector.append(calculate_distance(points[45], points[36])) # Eye outer width
#         feature_vector.append(calculate_distance(points[42], points[39])) # Eye inner width
#         feature_vector.append(calculate_distance(points[39], points[36])) # Left eye width
#         feature_vector.append(calculate_distance(points[45], points[42])) # Right eye width
#         feature_vector.append(calculate_distance(points[45], points[27])) # Right face width
#         feature_vector.append(calculate_distance(points[36], points[27])) # Left face width
#         feature_vector.append(calculate_distance(points[35], points[31])) # Nose width
#         feature_vector.append(calculate_distance(points[54], points[48])) # Mouth width
#         feature_vector.append(calculate_distance(points[33], points[27])) # Nose height
#         feature_vector.append(calculate_distance(points[48], points[36])) # Cheek height left
#         feature_vector.append(calculate_distance(points[54], points[45])) # Cheek height right
#         feature_vector.append(calculate_distance(points[63], points[33])) # Upper lip height
#         feature_vector.append(calculate_distance(points[62], points[51])) # Upper lip height
#         feature_vector.append(calculate_distance(points[66], points[57])) # Lower lip height
#         features.append(feature_vector)
#     return np.array(features)

# def get_landmarks(image):
#     with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
#         results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         if not results.multi_face_landmarks:
#             return None
#         landmarks = results.multi_face_landmarks[0]
#         points = [(lm.x, lm.y) for lm in landmarks.landmark]
#         return points
    

# image_paths = []

# landmarks_list = []
# for image_path in image_paths:
#     image = cv2.imread(image_path)
#     if image is None:
#         continue
#     landmarks = get_landmarks(image)
#     if landmarks:
#         landmarks_list.append(landmarks)

# features = extract_features(landmarks_list)

# X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, y_pred))
# print('Classification Report:')
# print(classification_report(y_test, y_pred))

import numpy as np
import pandas as pd
import cv2
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


mp_face_mesh = mp.solutions.face_mesh

def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
\
def extract_features(landmarks):
    features = []
    for points in landmarks:
        feature_vector = []
        # Example: Outer mouth distance calculation (points 48 to 59)
        feature_vector.append(calculate_distance(points[48], points[59]))
        # Add other feature calculations as per Tables 1 and 2
        feature_vector.append(calculate_distance(points[60], points[67])) # Inner mouth
        feature_vector.append(calculate_distance(points[17], points[21])) # Right eyebrow
        feature_vector.append(calculate_distance(points[22], points[26])) # Left eyebrow
        feature_vector.append(calculate_distance(points[36], points[41])) # Right eye
        feature_vector.append(calculate_distance(points[42], points[47])) # Left eye
        feature_vector.append(calculate_distance(points[27], points[34])) # Nose
        feature_vector.append(calculate_distance(points[0], points[16]))  # Jaw
        # Add width calculations
        feature_vector.append(calculate_distance(points[0], points[16]))  # Forehead width
        feature_vector.append(calculate_distance(points[45], points[36])) # Eye outer width
        feature_vector.append(calculate_distance(points[42], points[39])) # Eye inner width
        feature_vector.append(calculate_distance(points[39], points[36])) # Left eye width
        feature_vector.append(calculate_distance(points[45], points[42])) # Right eye width
        feature_vector.append(calculate_distance(points[45], points[27])) # Right face width
        feature_vector.append(calculate_distance(points[36], points[27])) # Left face width
        feature_vector.append(calculate_distance(points[35], points[31])) # Nose width
        feature_vector.append(calculate_distance(points[54], points[48])) # Mouth width
        feature_vector.append(calculate_distance(points[33], points[27])) # Nose height
        feature_vector.append(calculate_distance(points[48], points[36])) # Cheek height left
        feature_vector.append(calculate_distance(points[54], points[45])) # Cheek height right
        feature_vector.append(calculate_distance(points[63], points[33])) # Upper lip height
        feature_vector.append(calculate_distance(points[62], points[51])) # Upper lip height
        feature_vector.append(calculate_distance(points[66], points[57])) # Lower lip height
        features.append(feature_vector)
    return np.array(features)

def get_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        points = [(lm.x, lm.y) for lm in landmarks.landmark]
        return points

def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  
            image_paths.append(os.path.join(directory, filename))
    return image_paths

image_directory = 'C:\\Users\\Naveen Raghav\\Desktop\\archive1\\AutismDataset\\consolidated\\Autistic'

image_paths = get_image_paths(image_directory)


landmarks_list = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        continue
    landmarks = get_landmarks(image)
    if landmarks:
        landmarks_list.append(landmarks)

features = extract_features(landmarks_list)

num_images = len(features)
labels = np.random.randint(0, 2, num_images)  

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred))
