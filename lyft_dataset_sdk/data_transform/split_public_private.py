"""
The script splits the data into public and private parts.

"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser("Convert annotations from Kaggle csv to Nuscences json.")
    arg = parser.add_argument
    arg("--gt_path", type=Path, help="Path to csv file with ground truth.", required=True)
    arg("--pred_path", type=Path, help="Path to the json file with predictions.", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    gt_df = pd.read_csv(args.gt_path)

    token2test_type = dict(zip(gt_df["Id"].values, gt_df["Usage"].values))

    with open(args.pred_path) as f:
        predictions = json.load(f)

    for test_type in gt_df["Usage"].unique():
        result = []
        for entry in tqdm(predictions):
            sample_token = entry["sample_token"]
            if test_type == token2test_type[sample_token]:
                result += [entry]

        print(args.pred_path.parent / f"{args.pred_path.stem}_{test_type}.json")

        with open(args.pred_path.parent / f"{args.pred_path.stem}_{test_type}.json", "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
