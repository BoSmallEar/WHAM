import argparse
import os
import pickle
import sys

import joblib
import numpy as np
from tqdm import tqdm


def main():
    out_list_train = []
    out_list_val = []
    root_folder = sys.argv[1]
    folders = os.listdir(f"{root_folder}/wham")
    f = (1280**2 + 720**2)**0.5
    cx = 0.5 * 1280
    cy = 0.5 * 720
    intrinsics_old = np.eye(3)
    intrinsics_old[0, 0] = f
    intrinsics_old[1, 1] = f
    intrinsics_old[0, 2] = cx
    intrinsics_old[1, 2] = cy

    for i, folder in tqdm(enumerate(folders), total=len(folders)):
        video_name = folder.split('/')[-1]
        if os.path.exists(f"{root_folder}/wham/{folder}/wham_output.pkl"):
            wham_result = joblib.load(
                f"{root_folder}/wham/{folder}/wham_output.pkl")
            for k in wham_result.keys():
                start_frame = wham_result[k]['frame_ids'][0]
                start_str = str(start_frame + 1).zfill(6)
                image_path = f"{video_name}/{start_str}.jpg"
                out_dict = {}

                verts = wham_result[k]['verts']
                verts = verts[..., None]
                verts_2d = intrinsics_old @ verts
                verts_2d = verts_2d[..., 0]
                verts_2d = verts_2d / np.maximum(
                    verts_2d[..., 2:3],
                    np.ones_like(verts_2d[..., 2:3]) * 1e-3)
                verts_2d_x = verts_2d[..., 0]
                verts_2d_y = verts_2d[..., 1]
                verts_2d_x[verts_2d_x < 0] = 0
                verts_2d_x[verts_2d_x > 1280] = 1280
                verts_2d_y[verts_2d_y < 0] = 0
                verts_2d_y[verts_2d_y > 720] = 720
                verts_2d_x_min = np.min(verts_2d_x, axis=1)
                verts_2d_x_max = np.max(verts_2d_x, axis=1)
                verts_2d_y_min = np.min(verts_2d_y, axis=1)
                verts_2d_y_max = np.max(verts_2d_y, axis=1)

                # pad bbox by 10% on each side
                verts_2d_x_min -= (verts_2d_x_max - verts_2d_x_min) * 0.1
                verts_2d_x_max += (verts_2d_x_max - verts_2d_x_min) * 0.1
                verts_2d_y_min -= (verts_2d_y_max - verts_2d_y_min) * 0.1
                verts_2d_y_max += (verts_2d_y_max - verts_2d_y_min) * 0.1
                verts_2d_x_min[verts_2d_x_min < 0] = 0
                verts_2d_x_max[verts_2d_x_max > 1280] = 1280
                verts_2d_y_min[verts_2d_y_min < 0] = 0
                verts_2d_y_max[verts_2d_y_max > 720] = 720

                bbox_2d = np.stack([
                    verts_2d_x_min, verts_2d_x_max, verts_2d_y_min,
                    verts_2d_y_max
                ],
                                   axis=-1)
                out_dict["image"] = image_path
                out_dict["bbox_2d"] = bbox_2d
                out_dict["global_trans"] = wham_result[k]['trans_world']
                out_dict["local_trans"] = wham_result[k]['trans-offset']
                out_dict["betas"] = wham_result[k]['betas']
                out_dict["body_pose"] = wham_result[k][
                    'pose_world'][:, 3:].reshape(-1, 23, 3)
                out_dict["global_orient"] = wham_result[k]['pose_world'][:, :3]
                out_dict["local_orient"] = wham_result[k]['pose'][:, :3]
                # train on 80% scenes
                if np.isnan(out_dict["global_trans"]).any() or np.isnan(
                        out_dict["local_trans"]).any():
                    continue
                if i < len(folders) * 0.8:
                    out_list_train.append(out_dict)
                else:
                    out_list_val.append(out_dict)

    with open(f"{root_folder}/train.pkl", "wb") as f:
        pickle.dump(out_list_train, f)

    with open(f"{root_folder}/val.pkl", "wb") as f:
        pickle.dump(out_list_val, f)


if __name__ == '__main__':
    main()
