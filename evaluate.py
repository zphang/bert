import argparse
import json
import pandas as pd

from sklearn.metrics import matthews_corrcoef, f1_score
from scipy.stats import pearsonr, spearmanr

import run_classifier

PROCESSORS = {
    "cola": run_classifier.ColaProcessor,
    "sst": run_classifier.SstProcessor,
    "mrpc": run_classifier.MrpcProcessor,
    "stsb": run_classifier.StsbProcessor,
    "qqp": run_classifier.QqpProcessor,
    "mnli": run_classifier.MnliProcessor,
    "qnli": run_classifier.QnliProcessor,
    "rte": run_classifier.RteProcessor,
    "xnli": run_classifier.XnliProcessor,
    "snli": run_classifier.SnliProcessor,
    "bcs": run_classifier.BcsProcessor,
}
OUTPUT_MODES = {
    "cola": "classification",
    "sst": "classification",
    "mrpc": "classification",
    "stsb": "regression",
    "qqp": "classification",
    "mnli": "classification",
    "qnli": "classification",
    "rte": "classification",
    "xnli": "classification",
    "snli": "classification",
    "bcs": "classification",
}


def simple_accuracy(pred_srs, label_srs):
    return (pred_srs == label_srs).mean()


def acc_and_f1(pred_srs, label_srs):
    acc = simple_accuracy(pred_srs, label_srs)
    f1 = f1_score(y_true=label_srs, y_pred=pred_srs)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(pred_srs, label_srs):
    pearson_corr = pearsonr(pred_srs, label_srs)[0]
    spearman_corr = spearmanr(pred_srs, label_srs)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, pred_srs, label_srs):
    assert len(pred_srs) == len(label_srs)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(label_srs, pred_srs)}
    elif task_name == "sst":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    elif task_name == "mrpc":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "stsb":
        return pearson_and_spearman(pred_srs, label_srs)
    elif task_name == "qqp":
        return acc_and_f1(pred_srs, label_srs)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(pred_srs, label_srs)}
    else:
        raise KeyError(task_name)


def load_labels(task_name, data_dir):
    processor = PROCESSORS[task_name]()
    examples = processor.get_dev_examples(data_dir)
    label2idx = {label: num for (num, label) in enumerate(processor.get_labels())}
    label_srs = pd.Series([label2idx[example.label] for example in examples])
    return label_srs


def load_preds(task_name, pred_file_path):
    pred_df = pd.read_csv(pred_file_path, header=None, sep="\t")
    output_mode = OUTPUT_MODES[task_name]
    if output_mode == "classification":
        pred_srs = pred_df.idxmax(axis=1)
    elif output_mode == "regression":
        pred_srs = pred_df[:, 0]
    else:
        raise KeyError(output_mode)
    return pred_srs


def compute_metrics_from_paths(task_name, pred_file_path, task_data_dir):
    pred_srs = load_preds(task_name, pred_file_path)
    label_srs = load_labels(task_name, task_data_dir)
    return compute_metrics(task_name, pred_srs, label_srs)


def main():
    parser = argparse.ArgumentParser(description='evaluation')
    parser.add_argument('--task-name', required=True)
    parser.add_argument('--pred-file-path', required=True)
    parser.add_argument('--task-data-dir', required=True)
    parser.add_argument('--no-print', action="store_true")
    parser.add_argument('--output-path', required=False, default=None)
    args = parser.parse_args()
    metrics = compute_metrics_from_paths(args.task_name, args.pred_file_path, args.task_data_dir)
    if not args.no_print:
        print(metrics)
    if args.output_path is not None:
        with open(args.output_path, "w") as f:
            f.write(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
