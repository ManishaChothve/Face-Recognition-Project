Face Detection and Recognition System
Project Overview
This project is a real-time Face Detection and Recognition System developed using Python and OpenCV. It detects faces through a webcam and recognizes a trained person using a machine learning-based face recognition model.
Technologies Used
Python
OpenCV
Haar Cascade Classifier
LBPH Face Recognizer
NumPy
Features
Real-time face detection
Face recognition using webcam
Trained face model using face_model.yml
Displays the recognized person's name
Detects unknown faces
Project Structure
Face_Recognition_Project/
│
├── dataset/
├── Face_detection.py
├── Train_model.py
├── Face_recognition.py
├── face_model.yml
└── README.md
How to Run
Install Python and OpenCV.
Add training images to the dataset folder.
Run Train_model.py to train the face recognition model.
Run Face_recognition.py.
Allow camera access.
The system detects and recognizes the trained face.
Output
The system detects the face using the webcam and displays the recognized person's name on the screen.
Future Improvements
Add multiple person recognition
Improve recognition accuracy
Add attendance marking
Store recognition records in a database
Create a graphical user interface
Author
Manisha Chothve