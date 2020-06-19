"""
Runs inference on one image
"""

import cv2
from datetime import datetime
from matplotlib.pyplot import imshow, imsave

from apply_net import path_output, get_fashion_metadata
from detectron2.utils.visualizer import Visualizer, ColorMode

from iMaterialist2020.imaterialist.config import initialize_imaterialist_config, update_weights_outpath
from iMaterialist2020.imaterialist.evaluator import iMatPredictor


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


def predicted_image_show(path_image, predictor, fashion_metadata):
    """
    Visualize and save an image predicted using the given predictor and labelled with the fashion data.

    :param dict_test:
    :return:
    """
    time_stamp = datetime.now().isoformat().replace(":", "")

    im = cv2.imread(path_image)
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


def load_model_predict_image(path_image="/home/nasty/imaterialist2020/data/raw/test/0afb6b28d4583e470c7d0c52268272a7.jpg"):
    # cfg = setup(args)
    cfg = initialize_imaterialist_config()

    # Merge from TRAINED config file.
    cfg.merge_from_file("/home/nasty/imaterialist2020/iMaterialist2020/configs/exp05.yaml")
    update_weights_outpath(cfg, "/home/nasty/imaterialist2020/output/exp05/model_0109999.pth")

    # Set max input size
    cfg.INPUT.MAX_SIZE_TEST = 1024

    # Generate Predictor
    predictor = iMatPredictor(cfg)

    # This small set of data just to provide label.
    fashion_metadata = get_fashion_metadata()

    # Call the visualizer.
    predicted_image_show(path_image, predictor, fashion_metadata)