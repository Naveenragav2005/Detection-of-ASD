import cv2
import os


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


image_path ='C:\\Users\\Naveen Raghav\\Desktop\\archive1\\AutismDataset\\consolidated\\Autistic\\img.jpg'


if not os.path.isfile(image_path):
    raise FileNotFoundError(f"The file at path {image_path} does not exist.")

image = cv2.imread(image_path)


if image is None:
    raise ValueError(f"Failed to load image from path {image_path}.")


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)


cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

