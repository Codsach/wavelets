import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from utils import extract_dwt_features, extract_wavelet_packet_features
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight='balanced')
X = []
y = []

dataset_path = "dataset"

for label in os.listdir(dataset_path):
    class_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_name), 0)
        img = cv2.resize(img, (128,128))
        
        dwt_feat = extract_dwt_features(img)
        wp_feat = extract_wavelet_packet_features(img)
        
        features = np.concatenate([dwt_feat, wp_feat])
        
        X.append(features)
        y.append(label)

model = RandomForestClassifier()
model.fit(X, y)

with open("wavelet_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))