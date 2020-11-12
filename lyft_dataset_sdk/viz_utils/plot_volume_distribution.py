"""
The script takes path to the json of the type

[{'sample_token': 'aafe49574de38387ef78a64d7acdb719a83b636645d0a13891dfe1fbf157d847',
  'translation': [2038.1536289828764, 1109.4276702316238, -17.53994508677644],
  'size': [3.109, 10.435, 3.626],
  'rotation': [0.28950765881076357, 0.0, 0.0, 0.95717569729382],
  'name': 'other_vehicle'},
 {'sample_token': 'aafe49574de38387ef78a64d7acdb719a83b636645d0a13891dfe1fbf157d847',
  'translation': [2091.3524741907995, 1075.5431226501667, -18.821073395481434],
  'size': [1.838, 4.87, 1.628],
  'rotation': [0.27347522384875905, 0.0, 0.0, 0.9618790474591237],
  'name': 'car'}]

and creates png with histogram distribution.
"""

import argparse
import json

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

with open("lyft_dataset_sdk/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

class_names = config["class_names"]


def parse_args():
    parser = argparse.ArgumentParser("Convert annotations from Kaggle csv to Nuscences json.")
    arg = parser.add_argument
    arg("-i", "--input_file", type=str, help="Path to the input json file.", required=True)
    arg("-nr", "--num_rows", type=int, help="Number of rows.", default=3)
    arg("-nc", "--num_columns", type=int, help="Number of columns.", default=3)
    arg("-o", "--output_file", type=str, help="Path to the output image file.", required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_rows * args.num_columns < len(class_names):
        raise ValueError(
            f"Number of classes {len(class_names)} is larger than "
            f"num_rows {args.num_rows} and "
            f"num_columns {args.num_columns}."
        )

    with open(args.input_file) as f:
        df = pd.DataFrame(json.load(f))

    df["volume"] = df["size"].apply(np.prod)

    fig, axs = plt.subplots(args.num_rows, args.num_columns, figsize=(10, 10))

    for class_id, class_name in enumerate(class_names):
        x = class_id // args.num_rows
        y = class_id % args.num_rows

        data = df.loc[df["name"] == class_name, "volume"]

        axs[x, y].hist(data)
        axs[x, y].set_title(class_name, fontsize=15)

    fig.suptitle("Volume distribution", fontsize=20)

    plt.tight_layout()

    plt.savefig(args.output_file)


if __name__ == "__main__":
    main()
