import os
import cv2

# img_dir = 'OpticalFlow/movie/sky-timelapse/sky-timelapse_img/IAA95bBo2z0_640360'
# i = 0
# for subdir, dirs, files in os.walk(img_dir):
#        for file in files:
#            file_path = os.path.join(subdir, file)
#            if file_path.endswith(".jpg"):
#                img = cv2.imread(file_path)
#                img = cv2.resize(img, (250,141))
#                cv2.imwrite('OpticalFlow/movie/sky-timelapse/sky-timelapse_img/IAA95bBo2z0/%d.png' %i ,img)
#                i += 1

img = cv2.imread('Input/Images/3-VAM9owl-o_frames_00000001.jpg')
img = cv2.resize(img, (250,141))
cv2.imwrite('OpticalFlow/warped_image/adjacent/sky-timelapse/3-VAM9owl-o/origin/0.png',img)
