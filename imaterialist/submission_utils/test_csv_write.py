import csv
from typing import List
import pickle

def filter_csv_write(list_list_dict: List[List[dict]], path_csv):
    """
    Write the list of csv predictions into CSV.
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
            if item["EncodedPixels"] == "":
                continue
            else:
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

def /test_filter_csvwrite():
    # Load masks
    data = pickle.load(open("/home/dyt811/Git/cvnnig/data_imaterialist2020/2020-05-25T014759_NSM0.75Prediction/result_file.pkl", 'rb'))
    filter_csv_write(data, "2020-05-26T005749_csvBlank.csv")