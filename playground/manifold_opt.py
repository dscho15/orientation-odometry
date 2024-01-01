import numpy as np


def generate_random_3D_points(n: int = 512) -> np.ndarray:
    return np.random.rand(3, n)

def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def extrinsic_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t.reshape(3, 1)
    R = R.reshape(3, 3)
    return np.hstack((R, t))

def euler_angles_to_rotation_matrix(theta: np.ndarray) -> np.ndarray:
    """
    Convert euler angles to rotation matrix
    :param theta: 3x1 euler angles
    :return: 3x3 rotation matrix
    """
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(theta[1]) * np.cos(theta[2])
    R[0, 1] = -np.cos(theta[1]) * np.sin(theta[2])
    R[0, 2] = np.sin(theta[1])
    R[1, 0] = (
        np.cos(theta[0]) * np.sin(theta[2])
        + np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2])
    )
    R[1, 1] = (
        np.cos(theta[0]) * np.cos(theta[2])
        - np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2])
    )
    R[1, 2] = -np.sin(theta[0]) * np.cos(theta[1])
    R[2, 0] = (
        np.sin(theta[0]) * np.sin(theta[2])
        - np.cos(theta[0]) * np.sin(theta[1]) * np.cos(theta[2])
    )
    R[2, 1] = (
        np.sin(theta[0]) * np.cos(theta[2])
        + np.cos(theta[0]) * np.sin(theta[1]) * np.sin(theta[2])
    )
    R[2, 2] = np.cos(theta[0]) * np.cos(theta[1])
    return R

K = intrinsic_matrix(512, 512, 256, 256)
P1 = K @ extrinsic_matrix(euler_angles_to_rotation_matrix([0.1, -0.1, 0.1]), np.r_[-0.5, -0.5, 2])
P2 = K @ extrinsic_matrix(euler_angles_to_rotation_matrix([-0.1, 0.1, 0.1]), np.r_[-0.7, -0.3, 2])

# project points from 3D to image plane
pts = generate_random_3D_points()
p1 = P1 @ np.vstack((pts, np.ones(pts.shape[1])))
p2 = P2 @ np.vstack((pts, np.ones(pts.shape[1])))

p1 = p1[:2, :] / p1[2, :]  # normalize
p2 = p2[:2, :] / p2[2, :]  # normalize

# clamp to image size (1024 x 1024)
p1[0, :] = np.clip(p1[0, :], 0, 1024)
p1[1, :] = np.clip(p1[1, :], 0, 1024)

# mess with the points using random noise
sigma = 0.01 
p1 += np.random.normal(0, 1, p1.shape) * sigma
p2 += np.random.normal(0, 1, p2.shape) * sigma

from flask import Flask, render_template

app = Flask(__name__)
@app.route("/")
def index():
    points1 = p1.T.tolist()
    points2 = p2.T.tolist()
    return render_template("index.html", points1=points1, points2=points2)

app.run(debug=True)



