import numpy as np
import cv2
import math
from argparse import ArgumentParser
from dataset import KittyDataset
from camera import PinholeCamera
from misc import (get_extractor, get_matcher)
from visualizer_2d import visualize_pair

def parse_args():
    
    args = ArgumentParser(description="Visual Odometry")
    args.add_argument("--timestamps", type=str, default="datasets/kitty/00/times.txt")
    args.add_argument("--camera_intrinsics", type=str, default="datasets/kitty/00/calib.txt")
    args.add_argument("--camera_model", type=str, default="pinhole")
    args.add_argument("--images_dir", type=str, default="datasets/kitty/00/image_2")
    args.add_argument("--feature_extractor", type=str, default="sift")
    args.add_argument("--matcher", type=str, default="ratio")
    args.add_argument("--output_dir", type=str, default="output")
    
    return args.parse_known_args()[0]

if __name__ == "__main__":
    
    args = parse_args()
    
    dataset = KittyDataset(args.images_dir, 
                           args.camera_intrinsics,
                           args.timestamps)
    
    # get camera model
    h, w = dataset.get_hw
    fx, fy, cx, cy = dataset.get_raw_extrinsics
    cam = PinholeCamera(h, w, cx, cy, fx, fy)
    
    # get feature extractor
    extractor = get_extractor(args.feature_extractor)
    
    # get matcher
    matcher = get_matcher(args.matcher)
    
    for i in range(len(dataset) - 1):
        
        # read images
        img1 = dataset[i]
        img2 = dataset[i + 1]
        
        # extract features
        kp1, desc1 = extractor(img1)
        kp2, desc2 = extractor(img2)
        
        # match features
        matches = matcher(desc1, desc2)
        
        # get matched keypoints
        kp1_matched = np.array([kp1[0] for m in matches])
        kp2_matched = np.array([kp2[1] for m in matches])
        
        # get camera pose
        E, mask = cv2.findEssentialMat(kp1_matched, 
                                       kp2_matched, 
                                       cameraMatrix=cam.intrinsics, 
                                       method=cv2.RANSAC, 
                                       prob=0.999, 
                                       threshold=1.0)
        
        # mask out outliers
        matches = matches[mask.ravel() == 1]

        visualize_pair(img1, img2, kp1, kp2, matches)
        
        _, R, t, mask = cv2.recoverPose(E, kp1_matched, kp2_matched, 
                                        cameraMatrix=cam.intrinsics, mask=mask)
        
        print(mask.shape)
        
        
        
        exit()
        
        # # compute relative pose
        # R_rel, t_rel = cam.relative_pose(R, t)
        
        # # compute absolute pose
        # R_abs, t_abs = cam.absolute_pose(R_rel, t_rel)
        
        # # compute relative translation
        # t_rel = np.linalg.norm(t_rel)
        
        # # compute absolute translation
        # t_abs = np.linalg.norm(t_abs)
        
        # # compute relative rotation
        # R_rel = np.rad2deg(np.arccos((np.trace(R_rel) - 1) / 2))
        
        # # compute absolute rotation
        # R_abs = np.rad2deg(np.arccos((np.trace(R_abs) - 1) / 2))
        
        # print(f"Relative Rotation: {R_rel} degrees")
        # print(f"Relative Translation: {t_rel} meters")
        # print(f"Absolute Rotation: {R_abs} degrees")
        # print(f"Absolute Translation: {t_abs} meters")
        # print("-" * 80) 