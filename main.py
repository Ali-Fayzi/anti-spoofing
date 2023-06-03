import os
import cv2
import numpy as np
import argparse
from src.anti_spoofing_predict import AntiSpoofPredict
from src.utils import parse_model_name , CropImage
def check_image(image):
    height, width = image.shape[:2]
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True
    
def detect_face(img, faceCascade):
    faces = faceCascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(110, 110))
    return faces

def webcam(model_dir,device_id,draw=True):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    cam=cv2.VideoCapture(0)
    cascPath = r"./src//opencv_haarcascade/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    while True:
        success, image = cam.read()
        if success:
            image = cv2.flip(image, 1)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_bbox = detect_face(img_gray, faceCascade)
            if len(image_bbox) > 0:
                image_bbox = image_bbox[0]
                prediction = np.zeros((1, 3))
                for model_name in os.listdir(model_dir):
                    h_input, w_input, _ , scale = parse_model_name(model_name)
                    param = {
                        "org_img": image,
                        "bbox": image_bbox,
                        "scale": scale,
                        "out_w": w_input,
                        "out_h": h_input,
                        "crop": True,
                    }
                    if scale is None:
                        param["crop"] = False
                    img = image_cropper.crop(**param)
                    prediction += model_test.predict(img, os.path.join(model_dir, model_name))
                label = np.argmax(prediction)
                score = prediction[0][label]/2
                if label == 1:
                    result_text = "RealFace Score: {:.2f}".format(score)
                    color = (0, 255, 0)
                else:
                    result_text = "FakeFace Score: {:.2f}".format(score)
                    color = (0, 0, 255)
                
                if draw:
                    cv2.rectangle(image,(image_bbox[0], image_bbox[1]),(image_bbox[0] + image_bbox[2], image_bbox[1]+image_bbox[3]),color, 2)
                    cv2.putText(image,result_text,(image_bbox[0], image_bbox[1] - 5),cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
            cv2.imshow("Anti-Spoofing", image)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                cv2.destroyAllWindows()
                cam.release()
                break       
        else:
            cv2.destroyAllWindows()
            cam.release()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anti Spoofing")
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./src/checkpoints",
        help="model_lib used to test")
    
    parser.add_argument(
        "--draw",
        type=bool,
        default=True,
        help="Draw Bounding Box Rectangle")
    args = parser.parse_args()
    webcam( args.model_dir, args.device_id,args.draw)
