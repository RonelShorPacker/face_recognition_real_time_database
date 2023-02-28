import numpy as np
import cv2
import pickle
import face_recognition as fr
import cvzone
import os

def main():
    with open('encodings.pickle', 'rb') as handle:
        img_dict_encode = pickle.load(handle)
    # cap = cv2.VideoCapture(0)
    while True:
        # suc, image = cap.read()
        image = cv2.imread("img.png")
        img_small = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25)
        img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        faces_locs = fr.face_locations(img_small)
        encoded_faces = fr.face_encodings(img_small, faces_locs)

        for encoded_face, face_loc in zip(encoded_faces, faces_locs):
            match = fr.compare_faces(list(img_dict_encode.values()), encoded_face)
            dists = fr.face_distance(list(img_dict_encode.values()), encoded_face)
            index = np.argmin(dists)
            if match[index]:
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                bbox = x1, y1, x2-x1, y2 - y1
                cvzone.cornerRect(image, bbox, rt=0)
                cv2.putText(image, f'{os.path.splitext(list(img_dict_encode.keys())[index])[0]}', (x1, y1 - 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0))
        cv2.imshow("Face", image)
        cv2.waitKey(1)


main()