import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
hat_img, glasses_img, dog_filter_img = [cv2.imread(f'./snap_image/{img_name}.png') for img_name in ['hat', 'glasses', 'dog']]

def put_filter(filter_img, img, x, y, w, h):
    filter_mask = cv2.resize(filter_img, (w, h))
    img[y:y+h, x:x+w] = np.where(filter_mask < 235, filter_mask, img[y:y+h, x:x+w])
    return img

def put_hat_and_glasses(img, x, y, w, h):
    face_height, face_width = h, w
    hat_width = face_width + 1
    hat_height = int(0.50 * face_height) + 1

    hat_mask = cv2.resize(hat_img, (hat_width, hat_height))
    glasses_mask = cv2.resize(glasses_img, (hat_width, hat_height))

    img[y - int(0.40 * face_height):y - int(0.40 * face_height) + hat_height, x:x+hat_width] = np.where(hat_mask < 235, hat_mask, img[y - int(0.40 * face_height):y - int(0.40 * face_height) + hat_height, x:x+hat_width])
    img[y - int(-0.20 * face_height):y - int(-0.20 * face_height) + hat_height, x:x+hat_width] = np.where(glasses_mask < 235, glasses_mask, img[y - int(-0.20 * face_height):y - int(-0.20 * face_height) + hat_height, x:x+hat_width])

    return img

choice = input('Enter your choice filter to launch: 1="put hat & glasses", any other number="put dog filter": ')
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.19, 7)

    for (x, y, w, h) in faces:
        if choice == '1':
            img = put_hat_and_glasses(frame, x, y, w, h)
        else:
            img = put_filter(dog_filter_img, frame, x, y, w, h)

    cv2.imshow("Snap Filters", img)
    if cv2.waitKey(1) == ord('e'):
        break

webcam.release()
cv2.destroyAllWindows()
