from argparse import ArgumentParser
from dataset import KittyDataset
from core.camera.camera import PinholeCamera
from misc import get_extractor, get_matcher
from visual_odometry import VisualOdometry


def parse_args():
    args = ArgumentParser(description="Visual Odometry")
    args.add_argument("--timestamps", type=str, default="datasets/kitty/00/times.txt")
    args.add_argument(
        "--camera_intrinsics", type=str, default="datasets/kitty/00/calib.txt"
    )
    args.add_argument("--camera_model", type=str, default="pinhole")
    args.add_argument("--images_dir", type=str, default="datasets/kitty/00/image_2")
    args.add_argument("--feature_extractor", type=str, default="sift")
    args.add_argument("--matcher", type=str, default="ratio")
    args.add_argument("--output_dir", type=str, default="output")

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

    visual_odom = VisualOdometry(cam, extractor, matcher)

    for i in range(len(dataset) - 1):
        visual_odom.process_frame(i, dataset[i])
