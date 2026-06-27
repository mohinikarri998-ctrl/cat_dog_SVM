import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ===========================
# Dataset Paths
# ===========================
cat_path = "dataset/train/cats"
dog_path = "dataset/train/dogs"

data = []
labels = []
original_images = []

# ===========================
# Function to Load Images
# ===========================
def load_images(folder, label):
    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        image = cv2.imread(path)

        if image is None:
            continue

        # Store original image for display
        original = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_images.append(original)

        # Resize for training
        resized = cv2.resize(image, (128, 128))

        data.append(resized.flatten())
        labels.append(label)

# Load Cats (Label = 0)
load_images(cat_path, 0)

# Load Dogs (Label = 1)
load_images(dog_path, 1)

# ===========================
# Convert to NumPy Arrays
# ===========================
data = np.array(data)
labels = np.array(labels)
original_images = np.array(original_images, dtype=object)

print("Total Images:", len(data))

# ===========================
# Split Dataset
# ===========================
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    data,
    labels,
    original_images,
    test_size=0.2,
    random_state=42
)

# ===========================
# Train SVM
# ===========================
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# ===========================
# Predict
# ===========================
y_pred = model.predict(X_test)

# ===========================
# Accuracy
# ===========================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# ===========================
# Display Test Images
# ===========================
plt.figure(figsize=(12,8))

num_images = min(6, len(img_test))

for i in range(num_images):

    plt.subplot(2,3,i+1)

    plt.imshow(img_test[i])

    actual = "Cat" if y_test[i] == 0 else "Dog"
    predicted = "Cat" if y_pred[i] == 0 else "Dog"

    plt.title(f"Actual: {actual}\nPredicted: {predicted}", fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()