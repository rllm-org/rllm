"""
Miscellaneous Utility Functions
"""

import random

import numpy as np
from PIL import Image


def get_image(image_path):
    with Image.open(image_path) as img:
        return img.convert("RGB")
        return np.array(img)


def pad_from_left(input_id_list, pad_token_id):
    max_len = max([len(input_id) for input_id in input_id_list])
    if len(input_id_list) == 1:
        max_len += random.randint(1, 100)  # add some randomness to the padding if the batch size is one, for better batch inference
    padded_input_ids = [[pad_token_id] * (max_len - len(input_id)) + input_id for input_id in input_id_list]
    return padded_input_ids


def merge_dicts(dict_list):
    merged_dict: dict[str, list] = {}
    for dictionary in dict_list:
        for key, value in dictionary.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    return merged_dict
