import cv2
import random
import numpy as np

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from imaterialist.data.datasets.make_dataset import load_dataset_into_dataframes
from imaterialist.data.datasets.rle_utils import rle_decode_string



# https://detectron2.readthedocs.io/tutorials/datasets.html
# https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
def convert_to_datadict(df_input):
    """
    :param df_input:
    :return:
    """
    dataset_dicts = []

    # Find the unique list of imageId, we will build the
    list_unique_ImageIds = df_input['ImageId'].unique().tolist()
    for idx, filename in enumerate(list_unique_ImageIds):

        record = {}
        
        # Convert to int otherwise evaluation will throw an error
        record['height'] = int(df_input[df_input['ImageId'] == filename]['Height'].values[0])
        record['width'] = int(df_input[df_input['ImageId'] == filename]['Width'].values[0])
        
        record['file_name'] = filename
        record['image_id'] = idx

        objs = []
        for index, row in df_input[(df_input['ImageId'] == filename)].iterrows():

            # Get binary mask
            mask = rle_decode_string(row['EncodedPixels'], row['Height'], row['Width'])

            # opencv 4.2+
            # Transform the mask from binary to polygon format
            contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)
            
            # opencv 3.2
            # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,
            #                                            cv2.CHAIN_APPROX_SIMPLE)
            
            segmentation = []

            for contour in contours:
                contour = contour.flatten().tolist()
                # segmentation.append(contour)
                if len(contour) > 4:
                    segmentation.append(contour) 

                    # Data for each mask
            obj = {
                'bbox': [row['x0'], row['y0'], row['x1'], row['y1']],
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': row['ClassId'],
                'attributes': row['AttributesIds'], # New key: attributes
                'segmentation': segmentation,
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def register_datadict(datadict_input, label_dataset:str = "sample_fashion_train"):
    """
    Register the data type with the Catalog function from Detectron2 code base.
    fixme: currently hard coded as sample_fashion_train sample_fashion_test
    """
    _, _, df_categories = load_dataset_into_dataframes()
    # Register the train and test and set metadata

    DatasetCatalog.register(label_dataset, lambda d=datadict_input: convert_to_datadict(d))
    MetadataCatalog.get(label_dataset).set(thing_classes=list(df_categories.name))