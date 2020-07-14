"""
Runs inference on one image
"""

import cv2
import pickle
import os
from pathlib import Path
from environs import Env
from datetime import datetime
from matplotlib.pyplot import imshow, imsave

from apply_net import get_fashion_metadata
from imaterialist.data.datasets.coco import register_datadict
from detectron2.utils.visualizer import Visualizer, ColorMode

from imaterialist.config import initialize_imaterialist_config, update_weights_outpath
from imaterialist.evaluator import iMatPredictor

env = Env()
env.read_env()

path_output = Path(env("path_output_images"))
path_data_interim = Path(env("path_interim"))
path_images_local = Path(env("path_images_local"))

def predicted_image_datadict(datadic_test, predictor, fashion_metadata):
    """
    Show 3 predicted images from the Fashion Dict (make sure it is the test set!)
    :param dict_test:
    :return:
    """
    import random
    from datetime import datetime
    from imaterialist.data.datasets.make_dataset import load_category_attributes
    seed = random.randint(0, 99999999)

    # Randomly Grab 9 samples, iterate through rows of them, convert to list of tuple.  :
    list_tuple = list(datadic_test.sample(n=50, random_state=seed).iterrows())
    _, list_datadic = zip(*list_tuple)

    for i, d in enumerate(list_datadic):
        time_stamp = datetime.now().isoformat().replace(":", "")

        im = cv2.imread(d["ImageId"])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Run through predictor
        outputs = predictor(im)

        # Visualize
        v = Visualizer(im[:, :, ::-1],
                       metadata=fashion_metadata,
                       scale=0.8,
                       instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                       )
        # Bring the data back to CPU before passing to Numpy to draw
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        imshow(v.get_image()[:, :, ::-1])

        imsave(f"{path_output}/{time_stamp}.png", v.get_image()[:, :, ::-1])


def predicted_image_show(path_image_file, predictor, fashion_metadata):
    """
    Visualize and save an image predicted using the given predictor and labelled with the fashion data.

    :param dict_test:
    :return:
    """
    time_stamp = datetime.now().isoformat().replace(":", "")

    for path in os.listdir(path_image_file):
        
        full_path = os.path.join(path_image_file, path)
        print(full_path)

    
        im = cv2.imread(full_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        # Run through predictor
        outputs = predictor(im)

        # Visualize
        v = Visualizer(im[:, :, ::-1],
                    metadata=fashion_metadata,
                    scale=0.8,
                    instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                    )
        # Bring the data back to CPU before passing to Numpy to draw
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        imshow(v.get_image()[:, :, ::-1])

        imsave(f"{path_output}/{path}.png", v.get_image()[:, :, ::-1])


def load_model_predict_image(path_image=path_images_local):
    # cfg = setup(args)
    cfg = initialize_imaterialist_config()

    # Merge from TRAINED config file.
    cfg.merge_from_file("/home/julien/data-science/kaggle/imaterialist/configs/exp06.yaml")
    update_weights_outpath(cfg, "/home/julien/data-science/kaggle/imaterialist/output/exp03/model_0109999.pth")

    # Set max input size
    cfg.INPUT.MAX_SIZE_TEST = 1024

    # Generate Predictor
    predictor = iMatPredictor(cfg)

    datadict_val = pickle.load(open(path_data_interim / 'imaterialist_test_multihot_n=100.p', 'rb'))
    register_datadict(datadict_val, "sample_fashion_test")

    # This small set of data just to provide label.
    fashion_metadata = get_fashion_metadata()

    # Call the visualizer.
    predicted_image_show(path_image, predictor, fashion_metadata)

if __name__ == '__main__':
    load_model_predict_image()

