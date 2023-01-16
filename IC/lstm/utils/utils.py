""" misc utilities for Image Captioning."""

# imports
import json
import logging
import os


# ------------------------------------------------------------------------------

# save json data given specific location

def save_json(json_data, filename, data_directory, extension=".json"):
    # create directory if it doesn't exist
    # if required, display/log

    os.makedirs(data_directory, exist_ok=True)

    # write data to file
    json_filepath = os.path.join(data_directory, filename + extension)

    with open(json_filepath, "w+") as json_file:
        json_file.write(json.dumps(json_data))

    # print("data written at...")
    return


# ------------------------------------------------------------------------------


def init_logger(log_file_path: str = None):
    """Initialize logger"""
    if log_file_path is not None and log_file_path != '':
        os.makedirs(os.path.split(log_file_path)[0], exist_ok=True)

    logging.basicConfig(
        filename=log_file_path,  # if None, does not write to file
        filemode='a',  # default is 'a'
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
