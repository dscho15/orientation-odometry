import numpy as np


class Frame(object):
    def __init__(self, id, camera, pose, timestamp) -> None:
        self.id = id
        self.camera = camera
        self.timestamp = timestamp
        self.pose = pose

    def __hash__(self) -> int:
        return self.id
