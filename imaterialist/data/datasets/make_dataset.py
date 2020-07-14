import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing

from imaterialist.data.datasets.rle_utils import rle_decode_string, rle2bbox

from environs import Env

env = Env()
env.read_env()

# Get training dataframe
path_data = Path(env("path_raw"))
path_image = path_data / "train/"
path_data_interim = Path(env("path_interim"))

def load_category_attributes(path_data: Path = path_data):
    # Get label descriptions
    with open(path_data / 'label_descriptions.json', 'r') as file:
        label_desc = json.load(file)

    df_categories = pd.DataFrame(label_desc['categories'])
    df_attributes = pd.DataFrame(label_desc['attributes'])

    return df_attributes, df_categories


def load_dataset_into_dataframes(path_data: Path = path_data, n_cases: int = 0):
    """
    Get all the CSV from the competition into dataframes.
    """

    path_label = path_data / 'train.csv'
    df = pd.read_csv(path_label)

    # Just getting a smaller df to make the rest run faster
    if n_cases == 0:
        df = df.copy()
    elif n_cases != 0:
        df = df[:n_cases].copy()

    # Get label descriptions
    with open(path_data/'label_descriptions.json', 'r') as file:
        label_desc = json.load(file)

    df_categories = pd.DataFrame(label_desc['categories'])
    df_attributes = pd.DataFrame(label_desc['attributes'])

    return df, df_attributes, df_categories


def attr_str_to_list(df, df_attributes):
    '''
    Function that transforms DataFrame AttributeIds which are of type string into a 
    list of integers. Strings must be converted because they cannot be transformed into Tensors
    '''
    lb = preprocessing.LabelBinarizer()

    attribute_list = df_attributes.id.unique()
    attribute_list = np.sort(np.insert(attribute_list, 1, 999))

    lb.fit(attribute_list)


    # cycle through all the non NaN rows - NaN causes an error
    for index, row in df.iterrows():
        
        # Treating str differently than int
        if isinstance(row['AttributesIds'], str):
            
            # Convert each row's string into a list of strings             
            df['AttributesIds'][index] = row['AttributesIds'].split(',')
            
            # Convert each string in the list to int
            df['AttributesIds'][index] = [int(x) for x in df['AttributesIds'][index]]
                        
        # If int - make it a list of length 1
        if isinstance(row['AttributesIds'], int):
            df['AttributesIds'][index] = [999]
               
        df['AttributesIds'][index] = lb.transform(df['AttributesIds'][index]).sum(axis=0) 
        

def create_datadict(df_labels_masks, df_attributes):
    """
    Creates the data dictionary necessary for Detectron2 which incorporated the additional following information:
        ImageId
        x0
        y0
        x1
        y1
    """

    # Get image file path required for dict and add it to our data frame

    # Get only the first 50K labels, out of 333K labels.
    datedic_labels_masks = df_labels_masks.copy()  # df sample

    # Append ImageId information.
    datedic_labels_masks['ImageId'] = str(path_image) + "/" + datedic_labels_masks['ImageId'] + ".jpg"

    # Get bboxes for each mask with our helper function
    bboxes = [rle2bbox(c.EncodedPixels, (c.Height, c.Width)) for n, c in datedic_labels_masks.iterrows()]

    # Turn list into array for proper indexing
    bboxes_array = np.array(bboxes)

    # Add each x, y coordinate as a column
    datedic_labels_masks['x0'], datedic_labels_masks['y0'], datedic_labels_masks['x1'], datedic_labels_masks['y1'] = bboxes_array[:, 0], bboxes_array[:, 1], bboxes_array[:,2], bboxes_array[:, 3]
    
    datedic_labels_masks = datedic_labels_masks.astype({"x0": int, "y0": int, "x1":int, 'y1':int})

    #Replace NaNs from AttributeIds by 999
    datedic_labels_masks = datedic_labels_masks.fillna(999)
    
    # Turn attributes from string to list of ints with padding
    attr_str_to_list(datedic_labels_masks, df_attributes) 
    
    return datedic_labels_masks

def main(n_sample_size: int = 0, train_test_split: float = 0.8):
    """
    Runs data processing scripts to turn raw train.csv dataframe from (../raw) into
        cleaned dataframe ready to be used by our dataset_dict.
    :param n_sample_size:
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data_full, df_attributes, _ = load_dataset_into_dataframes(n_cases=500)
    datadic_full = create_datadict(data_full, df_attributes)

    # if n_sample_size not specified, use entire data set.
    # If too large, use entire data set.
    if n_sample_size == 0 or int(n_sample_size) > datadic_full.shape[0]:
        n_sample_size = len(datadic_full)


    # Arbitrary split in training / testing dataframes
    n_train: int = round(n_sample_size * train_test_split)
    n_test: int = n_sample_size - n_train
    datadic_train = datadic_full[:n_train].copy()
    datadic_val = datadic_full[-n_test:].copy()

    pickle.dump(datadic_train, open(path_data_interim / f'imaterialist_train_multihot_n={n_train}.p', "wb"))
    pickle.dump(datadic_val, open(path_data_interim / f'imaterialist_test_multihot_n={n_test}.p', "wb"))

    # # Saving to feather format - faster than pickle 
    # datadic_train.reset_index().to_feather(path_data_interim / f'imaterialist_train_multihot_n={n_train}.feather')
    # datadic_val.reset_index().to_feather(path_data_interim / f'imaterailist_test_multihot_n={n_test}.feather')
    
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Change size to create larger datasets
    main(500)
