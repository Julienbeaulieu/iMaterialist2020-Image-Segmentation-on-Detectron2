
import logging
import csv
import torch
import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from environs import Env
from pathlib import Path
from matplotlib.pyplot import imsave
from typing import Any, Dict, List

from detectron2.data.detection_utils import read_image
from detectron2.engine import default_argument_parser, launch
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import ColorMode, Visualizer

from iMaterialist2020.imaterialist.data.datasets.coco import register_datadict, MetadataCatalog
from iMaterialist2020.imaterialist.config import setup_prediction
from iMaterialist2020.imaterialist.evaluator import iMatPredictor
from iMaterialist2020.imaterialist.data.datasets.rle_utils_old import mask_to_KaggleRLE
from iMaterialist2020.imaterialist.data.datasets.rle_utils import mask_to_KaggleRLE_downscale


LOGGER_NAME = "apply_net"
logger = logging.getLogger(LOGGER_NAME)

env = Env()
env.read_env()

path_data_interim = Path(env("path_interim"))
path_test_data = Path(env("path_test"))
path_output = Path(env("path_output"))

class FileGen:
    '''
    Class that lazily builds a list of file_paths from a directory.
    This is done by returning a generator using a generator expression. 
    Helps not run into memory issues
    '''

    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        return (os.path.join(self.file_path, fname)
                for fname in os.listdir(self.file_path)
                if os.path.isfile(os.path.join(self.file_path, fname)))


def execute_on_outputs(entry: Dict[str, Any], outputs: Instances) -> List[dict]:
    """
    Parse instance from prediction to return a dict of the easier to read attributes.
    :param entry:
    :param outputs:
    :return:
    """

    image_fpath = entry["file_name"]
    logger.info(f"Processing {image_fpath}")

    # Get predicted classes from outputs
    pred_classes = np.array(outputs.pred_classes.cpu().tolist())

    # Get attribute scores from outputs
    attr_scores = np.array(outputs.attr_scores.cpu())

    # Keep only attributes with a score > 0.5
    attr_filter = attr_scores > 0.5

    # Get the index of attributes where the score is > 0. Each item in the list
    # corresponds to the predicted attributes for one instance
    # attr_filtered = [np.array(torch.where(attr_filter[i])[0].to("cpu")) for i in range(len(attr_filter))]
    attr_filtered = [np.where(attr_filter[i])[0] for i in range(len(attr_filter))]


    # Get masks from outputs
    has_mask = outputs.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset

        # Old Non-Union RLE
        # rles = [
        #     mask_to_KaggleRLE(mask) for mask in outputs.pred_masks.cpu()
        # ]

        # New Union RLE
        rles = refine_masks(outputs.pred_masks.cpu())



        # Uncomment following code to encode the masks to compressed RLE 
        # instead of uncompressed RLE like above: 

        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        # rles = [
        #     mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        #     for mask in outputs.pred_masks.cpu()
        # ]
        # for rle in rles:
        #     # "counts" is an array encoded by mask_util as a byte-stream. Python3's
        #     # json writer which always produces strings cannot serialize a bytestream
        #     # unless you decode it. Thankfully, utf-8 works out (which is also what
        #     # the pycocotools/_mask.pyx does).
        #     rle["counts"] = rle["counts"].decode("utf-8")
    results = []

    # Cycle each instance of an image
    for k in range(len(outputs)):
        # Attribute 294 is the category for empty attributes so we make the tensor empty
        # if it contains 294
        if 294 in attr_filtered[k]:
            # Must be sent to CPU
            attr_filtered[k] = np.array(torch.tensor([], device='cuda:0').cpu())


        # per Kaggle requirement.
        attributes_sorted = get_attribute_ids(list(attr_filtered[k]))
        # Get image ID from full path string
        image_id = Path(image_fpath).stem
        class_id = str(pred_classes[k])
        if has_mask:
            result = {"ImageId": image_id,
                      "EncodedPixels": rles[k],
                      "ClassId": class_id,
                      "AttributesIds": attributes_sorted,  # attribute IDs must be comma separated and sorted.
                      }
            # Encoded Pixels MUST be SPACE separated
        else:
            result = {"ImageId": image_id,
                      "ClassId": str(pred_classes[k]),
                      "AttributesIds": attributes_sorted,  # attribute IDs must be comma separated and sorted.
                      }
        results.append(result)
    return results

def get_attribute_ids(att_ids: List[int]):
    """
    Get concatenated AttributesIds
    Args:
        att_ids: [int], list of apparel attributes
    Returns:
        att_ids: string, e.g. "2,10,55,91"
    """
    #  Source: https://www.kaggle.com/c/imaterialist-fashion-2020-fgvc7/overview/evaluation
    att_ids.sort()  # need to be sorted before concatenation
    return ','.join([str(a) for a in att_ids])

def export_results(result: Dict[str, Any]):
    """
    Take the results and write them out into CSV and PKL for upload and future review
    :param result:
    :return:
    """
    # Get and create the folder path name to ensure the subsequent write operation will be successful.
    out_fname = result["out_fname"]
    out_dir = os.path.dirname(out_fname)
    if len(out_dir) > 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Write out pickle
    path_pickle = path_output / f"{out_fname}.pkl"
    pickle_write(result, path_pickle)

    # Write out CSV
    path_csv = path_output / f"{out_fname}.csv"
    # Filter out the blank rows where no encoded pixel is given but class prediction is given...
    filter_csv_write(result["results"], path_csv)


def pickle_write(result, path_pickle):
    with open(path_pickle, "wb") as pickle_file:
        pickle.dump(result["results"], pickle_file)
        logger.info(f"Output saved to {path_pickle}")


def main(args, visualize=True):
    # datadic_train = pd.read_feather(path_data_interim / 'imaterialist_train_multihot_n=4000.feather')
    # datadic_val = pd.read_feather(path_data_interim / 'imterailist_val_multihot_n=1000.feather')
    
    # register_datadict(datadic_train, "sample_fashion_train")
    # register_datadict(datadic_val, "sample_fashion_test")

    # This small set of data just to provide label./home/nasty/imaterialis
    datadic_test = pd.read_feather(path_data_interim / 'imateralist_val_multihot_n=1000.feather')
    register_datadict(datadic_test, "sample_fashion_test")
    fashion_metadata = get_fashion_metadata()

    # This update the prediction weight and output path automatically.
    cfg = setup_prediction(args)

    # cfg must have (cfg.DATASETS.TEST[0])

    # Generate the predictor
    predictor = iMatPredictor(cfg)

    # Create a list of image files
    # Loop through all data and generate.
    file_list = FileGen(path_test_data)

    # Dictionary where we'll append the results of all images and instances
    all_results = {"results": [], "out_fname": 'result_file'}

    for file_name in file_list:
        img = read_image(file_name, format="BGR")  # predictor takes BGR format
        with torch.no_grad():

            if visualize:
                #==========================================
                # Call the visualizer, label and save data
                # =========================================

                #show_predicted_image(file_name, predictor, fashion_metadata)

                v = Visualizer(img[:, :, ::-1],
                               metadata=fashion_metadata,
                               scale=0.8,
                               instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                               )

                # Get outputs of model in eval mode
                outputs = predictor(img)["instances"]
                v = v.draw_instance_predictions(outputs.to("cpu"))
                time_stamp = datetime.now().isoformat().replace(":", "")
                name = Path(file_name).stem
                imsave(f"{path_output}/{name}_{time_stamp}.png", v.get_image()[:, :, ::-1])

            # Get results for image
            result = execute_on_outputs({"file_name": file_name, "image": img}, outputs)

            all_results["results"].append(result)

    # Dump all results to output path
    # Pkl and CSV
    export_results(all_results)
    print("Example prediction result: ")
    print(all_results["results"][0])  # verification


def filter_csv_write(list_list_dict: List[List[dict]], path_csv):
    """
    Write the list of csv predictions into CSV but omit the rows where EncodedPixels are empty
    :param list_dict:
    :param path_csv:
    :return:
    """
    # Flatten the two list.
    # Feturn item if they the encoded pixel is not  flat.
    flat_list = []
    # Iterate through image list.
    for sublist in list_list_dict:
        # Iterate through mask list
        for item in sublist:
            # If the EncodedPixel is empty, skip.
            # if item["EncodedPixels"] == "":
            #     continue
            # else:
            flat_list.append(item)

    # With blanks.
    # flat_list = [item for sublist in list_list_dict for item in sublist]

    # Source: https://stackoverflow.com/questions/3086973/how-do-i-convert-this-list-of-dictionaries-to-a-csv-file
    keys = flat_list[0].keys()
    with open(path_csv, 'w') as output_file:
        # quote char prevent dict_writer to quote string that contain separtor: ,
        # The attributes are separated by COMMA, and must be quoted, by using space,
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(flat_list)


def csv_write(list_list_dict: List[List[dict]], path_csv):
    """
    Write the list of csv predictions into CSV.
    :param list_dict:
    :param path_csv:
    :return:
    """
    # Flatten the two list.
    flat_list = [item for sublist in list_list_dict for item in sublist]


    # Source: https://stackoverflow.com/questions/3086973/how-do-i-convert-this-list-of-dictionaries-to-a-csv-file
    keys = flat_list[0].keys()
    with open(path_csv, 'w') as output_file:
        # quote char prevent dict_writer to quote string that contain separtor: ,
        # The attributes are separated by COMMA, and must be quoted, by using space,
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(flat_list)

    #path_submission = Path(path_csv).parent / "submission.csv"
    #reader = csv.reader(open(path_csv, "r"), skipinitialspace=True)
    #writer = csv.writer(open(path_submission, "w"), quoting=csv.QUOTE_NONE)
    #writer.writerows(reader)


def get_fashion_metadata():
    # datadic_val = pd.read_feather(path_data_interim / 'imaterailist_test_multihot_n=100.feather')
    # register_datadict(datadic_val, "sample_fashion_test")
    fashion_metadata = MetadataCatalog.get("sample_fashion_test")
    return fashion_metadata


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    args.eval_only = True
    args.config_file = "/home/nasty/imaterialist2020/iMaterialist2020/configs/exp05.yaml"
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
