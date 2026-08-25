import cv2
import os

# Face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

dataset_path = "dataset"

# Read all images from dataset
for file_name in os.listdir(dataset_path):

    image_path = os.path.join(dataset_path, file_name)

    image = cv2.imread(image_path)

    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected_faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in detected_faces:
        faces.append(gray[y:y+h, x:x+w])

        # Label 1 = Manisha
        labels.append(1)

print("Training started...")
print("Images used:", len(faces))

if len(faces) == 0:
    print("No faces detected in dataset.")
    print("Please use clear face photos.")
    exit()

# Train the model
recognizer.train(faces, __import__("numpy").array(labels))

# Save trained model
recognizer.save("face_model.yml")

print("Training completed successfully!")
print("Model saved as face_model.yml")