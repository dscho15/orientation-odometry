from argparse import ArgumentParser
from core.camera.camera import PinholeCamera
from dataset import KittyDataset, CustomDataset
from pathlib import Path

import numpy as np

from misc import get_extractor, get_matcher
from tqdm import tqdm

import pandas as pd
from visual_odometry import VisualOdometry
from vo_utils import convert_poses_to_quaternions


def parse_args():
    args = ArgumentParser(description="Visual Odometry")
    
    args.add_argument(
        "-d",
        "--dataset",
        default="custom",
        type=str
    )
    
    args.add_argument(
        "--camera_intrinsics",
        type=str,
        default="datasets/2011_09_26/calib_cam_to_cam.txt",
    )
    
    args.add_argument(
        "--images_dir",
        type=str,
        default="/mnt/nvme0/datasets/video_blue_sky/videoes_from_michael/vid7/FRAMES/C0042",
    )
    
    args.add_argument("--camera_model", type=str, default="pinhole")
    args.add_argument("--debug", action="store_true")
    args.add_argument("--feature_extractor", type=str, default="root_sift")
    args.add_argument("--matcher", type=str, default="standard")
    args.add_argument("--output_dir", type=str, default="output")
    args.add_argument("--seed", type=int, default=42)
    args.add_argument("-fov", "--fov_w_deg", type=float, default=60.0)
    args.add_argument("-n", "--n_imgs", type=int, default=200)
    args.add_argument("-v", "--visualize", action="store_true")

    return args.parse_known_args()[0]


if __name__ == "__main__":    
    args = parse_args()

    np.random.seed(args.seed)

    if args.dataset == "kitty":
        dataset = KittyDataset(args.images_dir, args.camera_intrinsics)
    elif args.dataset == "custom":
        dataset = CustomDataset(args.images_dir, n_imgs=args.n_imgs, fov_w_deg=args.fov_w_deg)
    else:
        raise Exception("Dataset not supported")

    extractor = get_extractor(args.feature_extractor)
    
    matcher = get_matcher(args.matcher)
    
    depth_model = None

    camera = PinholeCamera(*dataset.get_hw, *dataset.get_raw_extrinsics)
    
    visual_odom = VisualOdometry(camera, extractor, matcher, args.visualize)
    
    outliers, time_pose_est, time_feature_extract = 0, 0, 0
    pbar = tqdm(range(len(dataset) - 1))
    
    for i in pbar:
        
        (outliers, time_pose_est, time_feature_extract, average_movement) = visual_odom.process_frame(
            i, dataset[i]
        )
        
        pbar.set_description(
            "Outliers: {:.2f}, Pose Estimation Time Estimate: {:.2f}s, Feature Extraction Time Estimate: {:.2f}s, Median Matching Movement: {:.2f}".format(
                outliers, time_pose_est, time_feature_extract, average_movement
            )
        )
        
    quats = convert_poses_to_quaternions(visual_odom.pose_trajectory) # N, 4
    df = pd.DataFrame(quats, columns=['w', 'x', 'y', 'z'])
    df.to_csv(f'output/{Path(args.images_dir).stem}.csv', index=False)

    def slerp(q1: np.ndarray, q2: np.ndarray, t: float, epsilon: float = 1e-6):
        omega = np.arccos(np.dot(q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)))
        so = np.sin(omega)
        if so < epsilon:
            return q1
        return (np.sin((1.0 - t) * omega) / so) * q1 + (np.sin(t * omega) / so) * q2

    imgs = dataset.imgs
    orig_data = pd.read_csv(f"output/{Path(args.images_dir).stem}.csv")
    data = pd.DataFrame(columns=['img', 'w', 'x', 'y', 'z'])
    data['img'] = imgs     

    quats = []
    for i in range(len(orig_data)):
        quat = np.array(orig_data.iloc[i].values)
        quats.append(quat)
        
    diff = np.diff(quats, axis=0)
    indices = np.where(np.sum(diff, axis=1) != 0)[0]
    indices = indices + 1
    indices = np.insert(indices, 0, 0)

    intervals = []
    for i in range(len(indices)-1):
        idx1 = indices[i]
        idx2 = indices[i+1]
        intervals.append((idx1, idx2))

    for (idx1, idx2) in intervals[:-1]:
        
        q1 = orig_data.iloc[idx1, -4:].values
        q2 = orig_data.iloc[idx2, -4:].values
        
        for i, t in enumerate(np.linspace(0, 1, idx2 - idx1 + 1)):
            (w, x, y, z) = slerp(q1, q2, t)
            data.iloc[idx1 + i, 1:] = (w, x, y, z)
            
    # drop the frames that has nans in data frame
    data = data.dropna()        
    data.to_csv(f'output/{Path(args.images_dir).stem}.csv', index=False)