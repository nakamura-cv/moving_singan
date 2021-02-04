import cv2
import numpy as np
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', default='Comparison')
    parser.add_argument('--model_name', help='input mdoel dir', required=True)
    parser.add_argument('--input_name', help='input image name', required=True)
    opt = parser.parse_args()


    # Gifファイルを読み込む
    gif = cv2.VideoCapture('%s/%s/gif/%s' % (opt.input_dir, opt.model_name, opt.input_name))
    # スクリーンキャプチャを保存するディレクトリを生成
    dir = '%s/%s/%s' % (opt.input_dir, opt.model_name, opt.input_name[:-4])
    try:
        os.makedirs(dir)
    except OSError:
        pass

    i = 0
    while True:
        is_success, frame = gif.read()
        # ファイルが読み込めなくなったら終了
        if not is_success:
            break
        print(frame.shape[0])
        if (frame.shape[0]!=250): frame = cv2.resize(frame, (250,250))
        # cv2.imwrite('%s/%d.png' % (dir, i), frame)
        # cv2.imwrite('%s/%d.jpg' % (dir, i), frame)
        i += 1