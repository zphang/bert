import argparse
import os
import numpy as np
import pandas as pd

from evaluation.evaluate import OUTPUT_MODES, PROCESSORS

OUTPUT_NAMES = {
    "cola": "CoLA",
    "sst": "SST-2",
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "qqp": "QQP",
    "mnli": "MNLI-m",
    "mnli-mm": "MNLI-mm",
    "qnli": "QNLI",
    "rte": "RTE",
}


def read_tsv(tsv_path):
    return pd.read_csv(tsv_path, sep="\t", header=None)


def write_srs(srs, output_path):
    output_df = pd.DataFrame({"index": np.arange(len(srs)), "prediction": srs.values})
    return output_df.to_csv(output_path, sep="\t", index=False)


def main(input_base_path, output_base_path):
    for task_name in OUTPUT_NAMES:
        try:
            output_mode = OUTPUT_MODES[task_name]
            processor = PROCESSORS[task_name]()
            if task_name == "mnli-mm":
                input_path = os.path.join(input_base_path, "mnli", "mm_test_results.tsv")
            else:
                input_path = os.path.join(input_base_path, task_name, "test_results.tsv")
            df = read_tsv(input_path)
            if output_mode == "classification":
                label_dict = dict(enumerate(processor.get_labels()))
                output_srs = df.idxmax(axis=1).replace(label_dict)
            elif output_mode == "regression":
                output_srs = df[0]
            else:
                raise KeyError(output_mode)
            write_srs(
                output_srs,
                os.path.join(output_base_path, "{}.tsv".format(OUTPUT_NAMES[task_name]))
            )
        except FileNotFoundError:
            print("Skipping {}".format(task_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='glue')
    parser.add_argument('--input-base-path', required=True)
    parser.add_argument('--output-base-path', required=True)
    args = parser.parse_args()
    main(args.input_base_path, args.output_base_path)
