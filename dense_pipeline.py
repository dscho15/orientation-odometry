from numpy import ndarray as NDArray
import cv2
import numpy as np
from math import ceil
from sparse_pipeline import convert_poses_to_quaternions, lp_filter, compute_error_and_median, parse_args, PinholeCamera
from dataset import CustomDataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from visualizer_2d import visualize_pair

def load_imgs_using_multithreading(paths, n: int = 8):
    
    def load_img(path):
        img = np.array(Image.open(path).convert('L'))
        return img
    
    with ThreadPoolExecutor(n) as executor:
        futures = []
        for image in paths:
            futures.append(executor.submit(load_img, image))
    
    processed_images = []
    for future in tqdm(futures):
        processed_images.append(future.result())
        
    return processed_images

def detect_features(img: NDArray[np.uint8],
                    window_size: int,
                    num_grid_x: 64,
                    num_grid_y: 64,
                    features_per_cell: int = 20) -> NDArray[np.int32]:
    height, width = img.shape[:2]

    wh = ceil(window_size / 2)
    grid_x = np.linspace(
        wh, width - wh, num_grid_x + 1, dtype=np.int32
    )
    grid_y = np.linspace(
        wh, height - wh, num_grid_y + 1, dtype=np.int32
    )

    features = []
    for iy in range(num_grid_y):
        for ix in range(num_grid_x):
            window = img[
                grid_y[iy] : grid_y[iy + 1], grid_x[ix] : grid_x[ix + 1]
            ]

            kps = cv2.goodFeaturesToTrack(
                window, features_per_cell, 0.1, minDistance=10
            )
            if kps is None:
                continue

            kps = kps[:, 0, :]

            for i in range(len(kps)):
                pt = (
                    int(round(kps[i, 0] + grid_x[ix])),
                    int(round(kps[i, 1] + grid_y[iy])),
                )
                features.append(pt)

    return np.asarray(features, dtype=np.int32)

def match_feature(
    img: NDArray[np.uint8], 
    ref: NDArray[np.uint8], 
    xy: tuple[int, int],
    template_size: int,
    window_size: int
):
    x, y = tuple(map(int, xy))

    tw = int(template_size // 2)
    template = ref[y - tw : y + tw + 1, x - tw : x + tw + 1]

    ww = int(window_size // 2)
    window = img[y - ww : y + ww + 1, x - ww : x + ww + 1]

    corr = cv2.matchTemplate(window, template, cv2.TM_CCORR_NORMED)
    min_val, max_val, _, max_loc = cv2.minMaxLoc(corr)

    if max_val / min_val < 1.01:
        max_val = 0

    out_x = max_loc[0] - ww + tw + x
    out_y = max_loc[1] - ww + tw + y
    
    return (out_x, out_y), max_val

def match_pipeline(ref: NDArray, 
                   img: NDArray,
                   window_size: int = 61,
                   num_grid_x: int = 16,
                   num_grid_y: int = 16,
                   template_size: int = 15,
                   features_per_cell: int = 20,
                   dist_tsh: float = 20) -> tuple[NDArray]:    
    # Find good features in reference image
    ref_pts = detect_features(ref, window_size, num_grid_x, num_grid_y, features_per_cell)

    # Match reference features to target image
    img_pts = np.zeros(ref_pts.shape, dtype=np.int32)
    corr = np.zeros(len(ref_pts))
    
    for i in range(len(ref_pts)):
        p, c = match_feature(img, ref, ref_pts[i], template_size, window_size)
        img_pts[i, :] = p
        corr[i] = c

    # Remove matches with high distance or low correlation
    dists = np.sqrt(np.sum((ref_pts - img_pts) ** 2, axis=1))
    good = (corr >= 0.95) & (dists <= dist_tsh)
    ref_pts = ref_pts[good]
    img_pts = img_pts[good]
    
    return ref_pts, img_pts

args = parse_args()

window_size = 71
num_grid_x = 32
num_grid_y = 32
template_size = 31
features_per_cell = 30
dist_tsh = 20

np.random.seed(args.seed)

dataset = CustomDataset(args.images_dir, n_imgs=args.n_imgs, fov_w_deg=args.fov_w_deg)

camera = PinholeCamera(*dataset.get_raw_extrinsics)

pbar = tqdm(range(len(dataset)))

imgs = load_imgs_using_multithreading(dataset.imgs)
    
matches = []
for i in tqdm(range(1, len(imgs))):
    ref = imgs[i-1]
    img = imgs[i]
    ref_pts, img_pts = match_pipeline(ref, img)    
    matches.append((ref_pts, img_pts))
    
def compute_pose(kps1, kps2, camera):
        
    # F, mask = cv2.findFundamentalMat(kps1, kps2, method=cv2.FM_RANSAC, ransacReprojThreshold=0.01, confidence=0.9999, maxIters=10000)
    
    # kps1 = kps1[mask.ravel() == 1]
    # kps2 = kps2[mask.ravel() == 1]
        
    F, mask = cv2.findFundamentalMat(kps1, kps2, cv2.FM_LMEDS)
    
    
    kps1 = kps1[mask.ravel() == 1]
    kps2 = kps2[mask.ravel() == 1]
    
    E = camera.K.T @ F @ camera.K
    
    _, R, t, _ = cv2.recoverPose(E, kps1, kps2, focal=1, pp=(0., 0.))
    
    return R, t, E, kps1, kps2

rotms = []
for i, (kp1, kp2) in tqdm(enumerate(matches)):
    R, E, t, kp1, kp2 = compute_pose(kp1, kp2, camera)
    
    visualize_pair(imgs[i], imgs[i+1], kp1, kp2, None, args.output_dir, idx=i)
    
    angle = cv2.Rodrigues(R)[0]
    angle = cv2.norm(angle, cv2.NORM_L2)
    
    if angle > np.pi / 8:
        rotms.append(np.eye(3, 3))
    else:
        rotms.append(R)
    
ori_traj_new = [np.eye(3, 3)]
for (R) in rotms:
    ori_traj_new.append(ori_traj_new[-1] @ R)
    
quats = convert_poses_to_quaternions(ori_traj_new)
lp_filtered = lp_filter(quats, alpha=args.alpha, normalize=False)
lp_filtered = lp_filter(lp_filtered[::-1], alpha=args.alpha, normalize=True)[::-1]

orig_data = pd.DataFrame(quats, columns=['x', 'y', 'z', 'w'])
orig_data['img'] = dataset.imgs[:len(quats)]
orig_data.to_csv(f'output/{Path(args.images_dir).stem}.csv', index=False)

plt.subplot(411)
plt.plot(quats[:, 0])
plt.title('x')
plt.subplot(412)
plt.plot(quats[:, 1])
plt.title('y')
plt.subplot(413)
plt.plot(quats[:, 2])
plt.title('z')
plt.subplot(414)
plt.plot(quats[:, 3])
plt.title('w')
plt.show()