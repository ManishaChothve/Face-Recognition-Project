import cv2

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load trained model
recognizer.read("face_model.yml")

# Your name
name = "Manisha"

# Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi zala")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
     faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(100, 100)
)
    
    

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]

        # Recognition
        id, confidence = recognizer.predict(face)

        # Display name if confidence is good
        if confidence < 70:
            display_name = name
        else:
            display_name = "Unknown"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            display_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", frame)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()