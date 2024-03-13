import os
import os.path as osp

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar

from lib.vis.renderer import Renderer, get_global_cameras
from lib.utils.transforms import axis_angle_to_matrix
from lib.utils.colors import get_colors


def run_vis_on_demo(cfg, video, results, output_pth, smpl):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT)

    # create renderer with cliff focal length estimation
    focal_length = (width**2 + height**2)**0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)

    # build default camera
    default_R, default_T = torch.eye(3), torch.zeros(3)


    writer = imageio.get_writer(osp.join(output_pth, 'output.mp4'),
                                fps=fps,
                                mode='I',
                                format='FFMPEG',
                                macro_block_size=1)
    bar = Bar('Rendering results ...', fill='#', max=length)
    global_colors = get_colors()
    global_colors = global_colors / 255.
    frame_i = 0
    _global_R, _global_T = None, None
    # run rendering
    while (cap.isOpened()):
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()
        if frame_i==0:
            init_img = img.copy()

        # render onto the input video
        renderer.create_camera(default_R, default_T)
        for _id, val in results.items():
            # render onto the image
            frame_i2 = np.where(val['frame_ids'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            img = renderer.render_mesh(torch.from_numpy(
                val['verts'][frame_i2]).to(cfg.DEVICE),
                                       img,
                                       colors=global_colors[_id])

        writer.append_data(img)
        bar.next()
        frame_i += 1
    writer.close()


def run_vis_on_demo_global(cfg, video, result, output_pth, smpl, id):
    # to torch tensor
    tt = lambda x: torch.from_numpy(x).float().to(cfg.DEVICE)

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT)

    # create renderer with cliff focal length estimation
    focal_length = (width**2 + height**2)**0.5
    renderer = Renderer(width, height, focal_length, cfg.DEVICE, smpl.faces)

     
        
    global_output = smpl.get_output(
        body_pose=tt(result['pose_world'][:, 3:]),
        global_orient=tt(result['pose_world'][:, :3]),
        betas=tt(result['betas']),
        transl=tt(result['trans_world']) +
        tt(result['trans'][[0]]))
    verts_glob = global_output.vertices.cpu()
    result['verts_glob'] = verts_glob

        
    writer = imageio.get_writer(osp.join(output_pth, f'output_{id}.mp4'),
                                fps=fps,
                                mode='I',
                                format='FFMPEG',
                                macro_block_size=1)
    bar = Bar('Rendering results ...', fill='#', max=length)
    global_colors = get_colors()
    global_colors = global_colors / 255.
    frame_i = 0
    _global_R, _global_T = None, None

    # run rendering
    while (cap.isOpened()):
        flag, org_img = cap.read()
        if not flag: break
        img = org_img[..., ::-1].copy()
        if frame_i < result['frame_ids'][0]: 
            frame_i += 1
            continue
        if frame_i==result['frame_ids'][0]:
            img_glob = img.copy()

        # build default camera
        default_R, default_T = torch.eye(3), torch.zeros(3)
        
        # render onto the input video
        renderer.create_camera(default_R, default_T)
       
        # render onto the image
        frame_i3 = np.where(result['frame_ids'] == frame_i)[0]
        if len(frame_i3) == 0: 
            frame_i += 1
            continue
        frame_i3 = frame_i3[0]
        img_glob = renderer.render_mesh(result['verts_glob'][frame_i3].to(cfg.DEVICE),
                                img_glob,
                                colors=global_colors[id])

        try:
            img = np.concatenate((img, img_glob), axis=1)
        except:
            img = np.concatenate((img, np.ones_like(img) * 255), axis=1)

        writer.append_data(img)
        bar.next()
        frame_i += 1
    writer.close()
