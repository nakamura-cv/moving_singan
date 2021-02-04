import cv2
import numpy as np
import os
import pathlib
import matplotlib.pyplot as plt

import argparse

def make_heatmap(diff, i, args):

    heatdata = diff.sum(axis=2)/3
    # plt.imshow(np.clip(np.linalg.norm(diff, ord=2, axis=2), a_min=0, a_max=255), cmap='cividis',vmax=255, vmin=0)
    plt.imshow(heatdata, cmap='OrRd',vmax=255, vmin=0)
    if i==0:plt.colorbar()
    plt.tick_params(labelbottom="off", bottom="off")  # x軸の削除
    plt.tick_params(labelleft="off", left="off")  # y軸の削除
    plt.savefig('Heatmap/%s/%s/%d.png' %(args.model_name, args.image_name, i))


def calculate_score_given_paths(path1, path2, suffix, args):
    path1 = pathlib.Path(path1)
    files1 = list(path1.glob('*.%s' %suffix))
    path2 = pathlib.Path(path2)
    files2 = list(path2.glob('*.%s' %suffix))

    files1.sort()
    files2.sort()

    # values = []
    # Im_ind = []

    for i in range(len(files2)):
        # print('real: ', files1[i])
        # print('fake: ', files2[i])
        real = cv2.imread(str(files1[i]))
        print(str(files1[i]))
        fake = cv2.imread(str(files2[i]))
        print(str(files2[i]))
        diff = np.abs(real.astype(int)-fake.astype(int))

        # cv2.imwrite('Diff/ours/%d.jpg' %i, diff)
        make_heatmap(diff, i, args)
        print('%d: ' %i ,diff.mean())
        if i == 0:
            diff_sum = diff
        else:
            diff_sum = diff_sum + diff

    return diff_sum.mean()/len(files2)



if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='Comparison',type=str)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--image_name', type=str, required=True)
    parser.add_argument('--images_suffix', default='png', type=str, help='image file suffix')

    args = parser.parse_args()

    try:
        os.makedirs('Heatmap/%s/%s' % (args.model_name, args.image_name))
    except OSError:
        pass

    path1 = '%s/real/%s' % (args.base_dir, args.image_name)
    path2 = '%s/%s/%s,%s' % (args.base_dir, args.model_name, args.image_name,args.image_name)
    suffix = args.images_suffix



    recon_score = calculate_score_given_paths(path1,path2,suffix,args)

    recon_score = np.asarray(recon_score,dtype=np.float32)
    print('score: ', recon_score.mean())
