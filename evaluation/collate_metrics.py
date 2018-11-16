import argparse
import json
import os

TASK_NAMES = ["cola", "sst", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]


def collate_metrics(base_path):
    collated = {}
    for task_name in TASK_NAMES:
        try:
            with open(os.path.join(base_path, task_name, "val_metrics.txt"), "r") as f:
                collated[task_name] = json.loads(f.read())
        except FileNotFoundError:
            print("Skipping {}".format(task_name))
    try:
        with open(os.path.join(base_path, "mnli", "mm_val_metrics.txt"), "r") as f:
            collated["mnli-mm"] = json.loads(f.read())
    except FileNotFoundError:
        print("Skipping {}".format("mnli-mm"))
    return collated


def dict_get(dictionary, *keys):
    try:
        curr = dictionary
        for key in keys:
            curr = curr[key]
        return curr
    except KeyError:
        return -0.01


def format_row(collated_metrics):
    return ",".join(map(str, [
        dict_get(collated_metrics, "cola", "mcc") * 100,
        dict_get(collated_metrics, "sst", "acc") * 100,
        "",
        "",
        "",
        "",
        dict_get(collated_metrics, "qnli", "acc") * 100,
        dict_get(collated_metrics, "rte", "acc") * 100,
        56.3,
        "",
        dict_get(collated_metrics, "qqp", "f1") * 100,
        dict_get(collated_metrics, "qqp", "acc") * 100,
        "",
        dict_get(collated_metrics, "mrpc", "f1") * 100,
        dict_get(collated_metrics, "mrpc", "acc") * 100,
        "",
        dict_get(collated_metrics, "stsb", "pearson") * 100,
        dict_get(collated_metrics, "stsb", "spearmanr") * 100,
        "",
        dict_get(collated_metrics, "mnli", "acc") * 100,
        dict_get(collated_metrics, "mnli-mm", "acc") * 100,
    ]))


def main():
    parser = argparse.ArgumentParser(description='collate')
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--no-print', action="store_true")
    parser.add_argument('--no-output', action="store_true")
    parser.add_argument('--output-path', required=False, default=None)
    parser.add_argument('--row-output-path', required=False, default=None)
    args = parser.parse_args()
    collated_metrics = collate_metrics(args.base_path)
    formatted_metrics = format_row(collated_metrics)
    if not args.no_print:
        print(json.dumps(collated_metrics, indent=2))
        print()
        print(formatted_metrics)
    if not args.no_output:
        if args.output_path is None:
            output_path = os.path.join(args.base_path, "collated_metrics.txt")
        else:
            output_path = args.output_path
        with open(output_path, "w") as f:
            f.write(json.dumps(collated_metrics, indent=2) + "\n")

        if args.row_output_path is None:
            row_output_path = os.path.join(args.base_path, "row.txt")
        else:
            row_output_path = args.row_output_path
        with open(row_output_path, "w") as f:
            f.write(formatted_metrics + "\n")


if __name__ == "__main__":
    main()
