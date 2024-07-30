# copied from C/lef2024-checkthat-lab/task5/scorer/verification_scorer.py for ease of use during dev only
import jsonlines
from csv import writer
import numpy as np


def strict_f1(actual, predicted, actual_evidence, predicted_evidence, label):

    tp = 0
    fp = 0
    fn = 0
    for i in range(len(actual)):
        if actual[i] != "NOT ENOUGH INFO":
            if (actual[i] == label) & (
                (predicted[i] == label)
                & (bool(set(predicted_evidence[i]) & set(actual_evidence[i])) == True)
            ):
                tp = tp + 1
            elif (actual[i] != label) & (predicted[i] == label):
                fp = fp + 1
            elif (actual[i] == label) & (
                (predicted[i] == label)
                & (bool(set(predicted_evidence[i]) & set(actual_evidence[i])) == False)
            ):
                fp = fp + 1
            elif (predicted[i] != label) & (actual[i] == label):
                fn = fn + 1
        else:
            if (actual[i] == label) & (predicted[i] == label):
                tp = tp + 1
            elif (actual[i] != label) & (predicted[i] == label):
                fp = fp + 1
            elif (predicted[i] != label) & (actual[i] == label):
                fn = fn + 1

    try:
        precision = tp / (tp + fp)
    except:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0
    return f1


def f1(actual, predicted, label):

    tp = 0
    fp = 0
    fn = 0
    for i in range(len(actual)):
        if (actual[i] == label) & (predicted[i] == label):
            tp = tp + 1
        elif (actual[i] != label) & (predicted[i] == label):
            fp = fp + 1
        elif (predicted[i] != label) & (actual[i] == label):
            fn = fn + 1

    try:
        precision = tp / (tp + fp)
    except:
        precision = 0
    try:
        recall = tp / (tp + fn)
    except:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except:
        f1 = 0
    return f1


def f1_macro(actual, predicted):
    # `macro` f1- unweighted mean of f1 per label
    return np.mean([f1(actual, predicted, label) for label in np.unique(actual)])


def f1_macro_strict(actual, predicted, actual_evidence, predicted_evidence):
    # `macro` f1- unweighted mean of macro-f1 per label
    return np.mean(
        [
            strict_f1(actual, predicted, actual_evidence, predicted_evidence, label)
            for label in np.unique(actual)
        ]
    )


def eval_run(pred_file, gold_file, out_file):
    gold_dict_labels = {}
    gold_dict_evidence = {}
    for line in jsonlines.open(gold_file):
        gold_dict_labels[line["id"]] = line["label"]
        temp_ev = []
        for ev in line["evidence"]:
            temp_ev.append(str(ev[1]))
        gold_dict_evidence[line["id"]] = temp_ev

    pred = [line for line in jsonlines.open(pred_file)]
    pred_labels = [line["predicted_label"] for line in pred]
    pred_evidence = []
    for line in pred:
        pred_instance = []
        for ev in line["predicted_evidence"]:
            pred_instance.append(str(ev[1]))
        pred_evidence.append(pred_instance)

    actual_labels = []
    actual_evidence = []
    for line in pred:
        actual_labels.append(gold_dict_labels[line["id"]])
        actual_instance = []
        for i in gold_dict_evidence[line["id"]]:
            actual_instance.append(i)
        actual_evidence.append(actual_instance)

    # compute macro-F1 and strict macro-F1
    macro_F1 = f1_macro(actual_labels, pred_labels)
    strict_macro_F1 = f1_macro_strict(
        actual_labels, pred_labels, actual_evidence, pred_evidence
    )

    print("Macro_F1", macro_F1)
    print("Strict Macro_F1", strict_macro_F1)

    result_list = [pred_file.split("/")[-1], macro_F1, strict_macro_F1]
    with open(out_file, "a") as f_object:
        writer_object = writer(f_object, delimiter="\t")
        writer_object.writerow(result_list)
        f_object.close()


def eval_run_custom(pred_file, gold_file, out_file):
    """
    basically the same, but without saving to file
    """
    gold_dict_labels = {}
    gold_dict_evidence = {}
    for line in jsonlines.open(gold_file):
        gold_dict_labels[line["id"]] = line["label"]
        temp_ev = []
        for ev in line["evidence"]:
            temp_ev.append(str(ev[1]))
        gold_dict_evidence[line["id"]] = temp_ev

    pred = [line for line in jsonlines.open(pred_file)]
    pred_labels = [line["predicted_label"] for line in pred]
    pred_evidence = []
    for line in pred:
        pred_instance = []
        for ev in line["predicted_evidence"]:
            pred_instance.append(str(ev[1]))
        pred_evidence.append(pred_instance)

    actual_labels = []
    actual_evidence = []
    for line in pred:
        actual_labels.append(gold_dict_labels[line["id"]])
        actual_instance = []
        for i in gold_dict_evidence[line["id"]]:
            actual_instance.append(i)
        actual_evidence.append(actual_instance)

    # compute macro-F1 and strict macro-F1
    macro_F1 = f1_macro(actual_labels, pred_labels)
    strict_macro_F1 = f1_macro_strict(
        actual_labels, pred_labels, actual_evidence, pred_evidence
    )
    return (macro_F1, strict_macro_F1)

import pyterrier as pt
import pyterrier.io as ptio
import pyterrier.pipelines as ptpipelines
from ir_measures import R, MAP    

if not pt.started():
    pt.init()

def eval_run_retrieval(pred_path, golden_path):
    golden = ptio.read_qrels(golden_path)
    pred = ptio._read_results_trec(pred_path)
    eval = ptpipelines.Evaluate(pred, golden, metrics = [R@5,MAP], perquery=False)
    return eval