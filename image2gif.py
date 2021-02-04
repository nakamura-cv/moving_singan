import imageio
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir', default='Comparison')
    parser.add_argument('--model_name', help='input mdoel dir', required=True)
    parser.add_argument('--input_name', help='input image name', required=True)
    parser.add_argument('--fps', type=int, help='frame per second', default=10)
    opt = parser.parse_args()

    img_dir = '%s/%s/%s' % (opt.input_dir, opt.model_name, opt.input_name)
    images = []
    for subdir, dirs, files in os.walk(img_dir):
       for file in sorted(files):
           file_path = os.path.join(subdir, file)
           if file_path.endswith(".png"):
                images.append(imageio.imread(file_path))
                print(file_path)
    imageio.mimsave('%s/%s/gif/%s.gif' % (opt.input_dir, opt.model_name, opt.input_name), images, fps=opt.fps)