import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

image_path = 'C:\\Users\\Naveen Raghav\\Desktop\\archive1\\AutismDataset\\consolidated\\Autistic\\img.jpg'
image = cv2.imread(image_path)

if image is None:
    raise ValueError(f"Failed to load image from path {image_path}.")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    results = face_detection.process(rgb_image)

if results.detections:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
               int(bboxC.width * iw), int(bboxC.height * ih)
        cv2.rectangle(image, bbox, (255, 0, 0), 2)

        with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            face_results = face_mesh.process(rgb_image)
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * iw)
                        y = int(landmark.y * ih)
                        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)


cv2.imshow('Detected Faces and Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
