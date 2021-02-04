import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
import numpy as np

import cv2


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo

    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-id', '--image_dir',default='Output/Editing/3-VAM9owl-o/start_scale=2,input=0.5')
    parser.add_argument('-id', '--image_dir',default='Output/Paint2image/moving-gif_00058/moving-gif_00058_2_out_quantized')
    parser.add_argument('-fd', '--flow_dir',default='OpticalFlow/estimated_flow/flow/adjacent/moving-gif/00130')
    parser.add_argument('-i', '--image_name', required=True)
    parser.add_argument('-f', '--flow_name', required=True)
    args = parser.parse_args()

    img = cv2.imread('%s/%s' % (args.image_dir, args.image_name))
    # print('img:',img.shape)
    # img = img[:,:128,:]
    # print('img:',img.shape)
    # img = cv2.resize(img, (int(img.shape[0]*2.0), int(img.shape[1]*2.0)))
    # print('img:',img.shape)
    # cv2.imwrite('warped_image/moving-gif/00130/0.png', img)
    img = torch.from_numpy(img.astype(np.float32))
    img = img.unsqueeze(0).permute(0,3,1,2)
    # print('img:',img.size())
    img = Variable(img).cuda()

    flow = cv2.readOpticalFlow('%s/%s' %(args.flow_dir, args.flow_name))
    flow = torch.from_numpy(flow.astype(np.float32))
    flow = flow.unsqueeze(0).permute(0,3,1,2)
    # print('flow:',flow.size())
    flow = Variable(flow).cuda()

    warped_img = warp(img, flow)
    warped_img = warped_img.squeeze(0).permute(1,2,0)
    # print('warped_img:',warped_img.size())
    warped_img = warped_img.cpu()
    warped_img = np.float32(warped_img)
    cv2.imwrite('Input/Paint/moving-gif_00058_%s.png' % ( args.flow_name[:-4]), warped_img)
    # cv2.imwrite('Input/Editing/3-VAM9owl-o/start_scale=2,input=0.5/%s.png' % ( args.flow_name[:-4]), warped_img)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-id', '--image_dir',default='data/Sintel_Video/scene42/42')
#     parser.add_argument('-fd', '--flow_dir',default='estimated_flow/flow/Sintel/42')
#     parser.add_argument('-i', '--image_name', required=True)
#     parser.add_argument('-f', '--first_frame', type=int, required=True)


#     args = parser.parse_args()

#     flow_list = []
#     for subdir, dirs, files in os.walk(args.flow_dir):
#         for file in files:
#             file_path = os.path.join(subdir, file)
#             if file_path.endswith(".flo"):
#                 flow_list.append(cv2.readOpticalFlow(file_path))

#     img = cv2.imread('%s/%s' % (args.image_dir, args.image_name))
#     img = torch.from_numpy(img.astype(np.float32))
#     img = img.unsqueeze(0).permute(0,3,1,2)
#     print('img:',img.size())
#     img = Variable(img).cuda()

#     for i in range(len(flow_list)):

#         flow = flow_list[i]
#         flow = torch.from_numpy(flow.astype(np.float32))
#         flow = flow.unsqueeze(0).permute(0,3,1,2)
#         print('flow:',flow.size())
#         flow = Variable(flow).cuda()

#         warped_img = warp(img, flow)
#         warped_img = warped_img.squeeze(0).permute(1,2,0)
#         print('warped_img:',warped_img.size())
#         warped_img = warped_img.cpu()
#         warped_img = np.float32(warped_img)
#         # cv2.imwrite('warped_image/Sintel/42/%s_warp.png' % (args.flow_name[:-4]), warped_img)
#         cv2.imwrite('warped_image/Sintel/42/%d-%d_warp.png' % (args.first_frame, (args.first_frame+i+1)) , warped_img)
