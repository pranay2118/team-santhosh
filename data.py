import os
import cv2

dataset = './data'
if not os.path.exists(dataset):
    os.makedirs(dataset)

number_of_classes = 3
dataset_size = 100

# Use 0 for the default camera, adjust if needed
cap = cv2.VideoCapture(0)  

for j in range(number_of_classes):
    class_dir = os.path.join(dataset, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        cv2.putText(frame, 'Ready? Press "Q"!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()