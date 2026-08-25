import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

image = cv2.imread("test_images/manisha.jpg")

if image is None:
    print("Photo not found")
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=8,
    minSize=(100, 100)
)

# फक्त सर्वात मोठा face निवडणे
if len(faces) > 0:

    x, y, w, h = max(
        faces,
        key=lambda r: r[2] * r[3]
    )

    face = gray[y:y+h, x:x+w]

    id, confidence = recognizer.predict(face)

    if confidence < 70:
        name = "Manisha"
    else:
        name = "Unknown"

    cv2.rectangle(
        image,
        (x, y),
        (x+w, y+h),
        (255, 0, 0),
        2
    )

    cv2.putText(
        image,
        name,
        (x, y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2
    )

else:
    print("No face detected")

cv2.imshow("Photo Recognition", image)

cv2.waitKey(0)
cv2.destroyAllWindows()