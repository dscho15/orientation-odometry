from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np

def get_extractor(extractor_name: str,
                  n_features: int = 10000):
    if extractor_name == "sift":
        from feature_extractor import SiftExtractor
        return SiftExtractor(n_features)
    elif extractor_name == "root_sift":
        from feature_extractor import RootSiftExtractor
        return RootSiftExtractor(n_features)
    elif extractor_name == "tomasi":
        from feature_extractor import GoodFeaturesToTrackExtractor
        return GoodFeaturesToTrackExtractor(n_features)
    elif extractor_name == "orb":
        from feature_extractor import OrbExtractor
        return OrbExtractor(n_features)
    else:
        raise NotImplementedError("Extractor not implemented")


def get_matcher(matcher_name: str):
    if matcher_name == "standard":
        from matcher import FeatureMatcher
        return FeatureMatcher()
    else:
        raise NotImplementedError("Matcher not implemented")
    
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