from argparse import ArgumentParser
from dataset import KittyDataset
from core.camera.camera import PinholeCamera
from misc import get_extractor, get_matcher
from visual_odometry import VisualOdometry
from visualizer_2d import visualize_quaternion_trajectory
from tqdm import tqdm


def parse_args():
    args = ArgumentParser(description="Visual Odometry")
    args.add_argument(
        "--timestamps",
        type=str,
        default="datasets/2011_09_26/2011_09_26_drive_0018_extract/image_02/timestamps.txt",
    )
    args.add_argument(
        "--camera_intrinsics",
        type=str,
        default="datasets/2011_09_26/calib_cam_to_cam.txt",
    )
    args.add_argument("--camera_model", type=str, default="pinhole")
    args.add_argument(
        "--images_dir",
        type=str,
        default="datasets/2011_09_26/2011_09_26_drive_0018_extract/image_02",
    )
    args.add_argument("--feature_extractor", type=str, default="orb")
    args.add_argument("--matcher", type=str, default="standard")
    args.add_argument("--output_dir", type=str, default="output")
    args.add_argument("-v", "--visualize", action="store_true")

    return args.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_args()

    dataset = KittyDataset(args.images_dir, args.camera_intrinsics, args.timestamps)

    frames = []

    # get camera model
    h, w = dataset.get_hw
    fx, fy, cx, cy = dataset.get_raw_extrinsics
    cam = PinholeCamera(h, w, cx, cy, fx, fy)
    extractor = get_extractor(args.feature_extractor)
    matcher = get_matcher(args.matcher)

    visual_odom = VisualOdometry(cam, extractor, matcher, args.visualize)
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
        
    visualize_quaternion_trajectory(visual_odom.pose_history)
