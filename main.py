import cv2


def main():
    cap = cv2.VideoCapture(0)
    while True:
        suc, image = cap.read()
        cv2.imshow("Face", image)
        cv2.waitKey(1)


main()