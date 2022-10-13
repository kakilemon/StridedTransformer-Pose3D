import sys
import argparse
import os 
import numpy as np
import torch
import glob
from tqdm import tqdm
import copy

sys.path.append(os.getcwd())
from model.strided_transformer import Model
from common.camera import *

def get_pose3D(keypoints, length, size):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.layers, args.channel, args.d_hid, args.frames = 3, 256, 512, 351
    args.stride_num = [3, 9, 13]
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/pretrained'
    args.n_joints, args.out_joints = 17, 17

    ## Reload 
    model = Model(args).cuda()

    model_dict = model.state_dict()
    model_paths = sorted(glob.glob(os.path.join(args.previous_dir, '*.pth')))
    for path in model_paths:
        if path.split('/')[-1][0] == 'n':
            model_path = path

    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)
    model.eval()

    ## input
    # keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']

    video_length = length

    ## 3D
    print('\nGenerating 3D pose...')
    res = []
    for i in tqdm(range(video_length)):
        img_size = size

        ## input frames
        start = max(0, i - args.pad)
        end =  min(i + args.pad, len(keypoints[0])-1)

        input_2D_no = keypoints[0][start:end+1]
        
        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')
        
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[ :, :, 0] *= -1
        input_2D_aug[ :, joints_left + joints_right] = input_2D_aug[ :, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        
        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        N = input_2D.size(0)

        ## estimation
        output_3D_non_flip, _ = model(input_2D[:, 0])
        output_3D_flip, _     = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :] 

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()

        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])
        res.append(post_out)
    return np.array(res)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoints', type=str, default='keypoints.npz', help='keypoints')
    parser.add_argument('--length', type=int, default=255, help='length')
    parser.add_argument('--sizex', type=int, default=1080, help='x')
    parser.add_argument('--sizey', type=int, default=1920, help='y')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    keypoints = np.load(args.keypoints, allow_pickle=True)['reconstruction']
    print(args.length, (args.sizex, args.sizey))
    out = get_pose3D(keypoints, args.length, (args.sizex, args.sizey))
    np.save('out', out)
