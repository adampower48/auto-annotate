"""
Extracts relevant information from a KITTI file

Output Headers:
    frame, object_id, object_type, xmin, ymin, xmax, ymax
"""


import pandas as pd
import os


# These can be changed
IMPORTANT_COLUMNS = [0, 1, 2, 6, 7, 8, 9]
COLUMN_MAPPINGS = {
    0: "frame",
    1: "object_id",
    2: "object_type",
    6: "xmin",
    7: "ymin",
    8: "xmax",
    9: "ymax",
}


def set_config(mappings_list):
    """
    mappings_list: a list of dictionary (key, value) pairs
    """

    global COLUMN_MAPPINGS, IMPORTANT_COLUMNS

    COLUMN_MAPPINGS = dict(mappings_list)
    IMPORTANT_COLUMNS = sorted(list(COLUMN_MAPPINGS.keys()))


def read_and_clean_kitti(filename):
    # Read, filter & rename columns
    df = pd.read_csv(filename, sep=" ", header=None, )
    df = df.iloc[:, IMPORTANT_COLUMNS].rename(columns=COLUMN_MAPPINGS)

    return df


def make_folder(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        # todo: handle permission errors?
        # print(f"{path} exists")
        pass
