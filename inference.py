import argparse
import os
import os.path as op
import csv

from tqdm import tqdm
from transformers import pipeline


def main(args):
    classifier = pipeline("text-classification", model=args.model_name)
    label2id = {"world": 0, "sport": 1, "business": 2, "sci-tech": 3}
    result = []

    with open(args.output_path, 'w') as f:
        f.write("ID,Class Index\n")

    with open(args.dataset_path, newline='') as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')

        for row in tqdm(reader):
            sample_id = row[0]
            text = ' '.join([row[1], row[2]])

            generated = classifier(text)
            generated = label2id[generated[0]["label"]] + 1
            
            with open(args.output_path, 'a') as f:
                f.write(f"{sample_id},{generated}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--model_name", type=str, default="distillbert-base-uncased")

    args = parser.parse_args()

    assert op.exists(args.dataset_path), "Dataset path doesn't exist"

    main(args)