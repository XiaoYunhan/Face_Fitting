import sys
import os
import dlib
import glob

dir_bill = "bill-clinton.jpg"
dir_hillary = "hillary-clinton.jpg"
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

for f in glob.glob(os.path.join(dir_bill)):
    img = dlib.load_rgb_image(f)
    win.clear_overlay()
    win.set_image(img)
    dets = detector(img, 1)
    

