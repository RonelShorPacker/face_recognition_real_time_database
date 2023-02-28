import os
import pickle
import cv2
import face_recognition as fr


def getEncodings(img_dict):
    encodings = []
    for img in img_dict.values():
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodings.append(encode)

    return encodings

def readDatabase():
    path = './Database/'
    img_dict = {}
    images_names = os.listdir(path)
    for name in images_names:
        img_dict[name] = cv2.imread(path+name)
    return img_dict

if __name__=="__main__":
    img_dict = readDatabase()
    img_dict_encode = getEncodings(img_dict)
    with open('encodings.pickle', 'wb') as handle:
        pickle.dump(img_dict_encode, handle, protocol=pickle.HIGHEST_PROTOCOL)