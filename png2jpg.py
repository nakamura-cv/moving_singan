import cv2
import numpy as np
import os
import pathlib

import argparse

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_dir', help='input image dir', required=True)
#     parser.add_argument('--input_name', help='input image name', required=True)
#     opt = parser.parse_args()

#     img_dir = '%s/%s' % (opt.input_dir, opt.input_name)
#     print(img_dir)
#     image = cv2.imread(img_dir)
#     image = cv2.resize(image, (250,250))
#     cv2.imwrite('Comparison/ours/jpg/moving-gif_00130,moving-gif_00130/%s.jpg' % opt.input_name[:-4], image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='Comparison',type=str)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--image_name', type=str, required=True)

    args = parser.parse_args()

    path = '%s/%s/%s' % (args.base_dir, args.model_name, args.image_name)
    path = pathlib.Path(path)
    files = list(path.glob('*.png'))
    files.sort()

    for i in range(len(files)):
        img = cv2.imread(str(files[i]))
        cv2.imwrite('%s/%d.jpg' %(path,i), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
