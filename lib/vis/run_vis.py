import os
import os.path as osp

import cv2
import torch
import imageio
import numpy as np
from progress.bar import Bar

from lib.vis.renderer import Renderer, get_global_cameras
from lib.utils.transforms import axis_angle_to_matrix, matrix_to_axis_angle
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

     
        
    ## suppose we want to use the 0th frame as the starting frame  
    START_FRAME = 0

    global_orient_source = axis_angle_to_matrix(tt(result['pose_world'][START_FRAME, :3]))
    global_orient_target = axis_angle_to_matrix(tt(result['pose'][START_FRAME, :3]))

    source_to_target_rotation =  global_orient_target @ global_orient_source.T
    global_orient= axis_angle_to_matrix(tt(result['pose_world'][START_FRAME:, :3]))
    global_orient = matrix_to_axis_angle(source_to_target_rotation @ global_orient)

    transl_source = tt(result['trans_world'][[START_FRAME]]) 
    transl_target = tt(result['trans'][[START_FRAME]])
    source_to_target_translation = transl_target.T - source_to_target_rotation@transl_source.T
    transl = tt(result['trans_world'][START_FRAME:])  
    transl = source_to_target_rotation @ transl.T + source_to_target_translation
    transl = transl.T


        
    global_output = smpl.get_output(
        body_pose=tt(result['pose_world'][START_FRAME:, 3:]),
        global_orient=global_orient,
        betas=tt(result['betas'][START_FRAME:]),
        transl= transl)
    
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
