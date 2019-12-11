import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="image to be classified")
    parser.add_argument("-m", "--model", type=str, help="model used for classification")

    return parser.parse_args()