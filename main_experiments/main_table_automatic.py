import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

from src.eval import compute_unchanged_score, compute_change_score
from src.utils import remove_duplicates, set_seed

rm_dup = False # filter test set; only changes in the first comma for CoDEx-M

set_seed(0)

def generate_table(dictionary, metrics, roundeds=2):
    df = []
    df_without_str = []

    for method, method_name in methods.items():
        df_row_1 = [method_name]
        df_row_2 = [f"COULDD-{method_name}"]

        df_without_str_row_1 = [method_name]
        df_without_str_row_2 = [f"COULDD-{method_name}"]

        for dataset, dataset_name in datasets.items():
            for metric, metric_name in metrics.items():

                mean_1 = round(dictionary[dataset][method]["original"][metric] * 100, roundeds)

                mean = round(np.mean(dictionary[dataset][method]["exp"][metric]) * 100, roundeds)
                std = round(np.std(dictionary[dataset][method]["exp"][metric]) * 100, roundeds)

                if mean - mean_1 > 0:
                    df_row_1.append(mean_1)
                    df_row_2.append(f"\\textbf{{{mean} $\pm$ {std}}}")
                else:
                    df_row_1.append(f"\\textbf{{{mean_1}}}")
                    df_row_2.append(f"{mean} $\pm$ {std}")

                df_without_str_row_1.append(mean_1)
                df_without_str_row_2.append(mean)

        df.append(df_row_1)
        df.append(df_row_2)

        df_without_str.append(df_without_str_row_1)
        df_without_str.append(df_without_str_row_2)

    chatgpt_row = ["gpt-3.5-turbo"]
    for dataset, dataset_name in datasets.items():
        for metric, metric_name in metrics.items():
            chatgpt_row.append(round(dictionary[dataset]["chatgpt"][metric] * 100, roundeds))
    df.append(chatgpt_row)
    df_without_str.append(chatgpt_row)

    columns = [("Method", "")] + [(dataset_name, metric_name) for dataset_name in datasets.values() for metric_name in
                                  metrics.values()]

    df = pd.DataFrame(df, columns=pd.MultiIndex.from_tuples(columns))
    df_without_str = pd.DataFrame(df_without_str, columns=pd.MultiIndex.from_tuples(columns))

    for column in df.columns[1:]:
        # find max and set the string underlined in df
        max_row = df_without_str[column].astype(float).argmax()
        df.loc[max_row, column] = f"\\underline{{{df.loc[max_row, column]}}}"

    return df


methods = {
    "rescal": "RESCAL",
    "transe": "TransE",
    "complex": "ComplEx",
    "conve": "ConvE",
    "tucker": "TuckER",
}

datasets = {
    "s": "CFKGR-CoDEx-S",
    "m": "CFKGR-CoDEx-M",
    "l": "CFKGR-CoDEx-L",
}
path_to_codex_folder = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
data_dir = os.path.join(path_to_codex_folder, 'results_test')

dictionary = {}

for dataset in datasets.keys():
    dictionary[dataset] = {}

    for method in methods.keys():
        pred_df = pd.read_csv(os.path.join(os.path.join(data_dir, f"codex-{dataset}", method, "early_stop_True_batches_uniform", f"test_predictions.csv")))
        if rm_dup is True:
            pred_df = remove_duplicates(pred_df)

        local_dict = {}
        accs, f1s, recalls, precisions, inferences, retentions = [], [], [], [], [], []
        for i in range(5):
            accs.append(accuracy_score(y_true=pred_df["expected_label"], y_pred=pred_df[f"couldd_preds_{i}_binary"]))
            f1s.append(f1_score(y_true=pred_df["expected_label"], y_pred=pred_df[f"couldd_preds_{i}_binary"]))
            recalls.append(recall_score(y_true=pred_df["expected_label"], y_pred=pred_df[f"couldd_preds_{i}_binary"]))
            precisions.append(
                precision_score(y_true=pred_df["expected_label"], y_pred=pred_df[f"couldd_preds_{i}_binary"]))

            inferences.append(
                compute_change_score(torch.tensor(pred_df["expected_label"].values), torch.tensor(pred_df[f"couldd_preds_{i}_binary"].values),
                                     torch.tensor(pred_df["original_label"].values))
            )
            retentions.append(compute_unchanged_score(
                torch.tensor(pred_df["expected_label"].values), torch.tensor(pred_df[f"couldd_preds_{i}_binary"].values), torch.tensor(pred_df["original_label"].values))
            )

        local_dict["acc"] = accs
        local_dict["f1"] = f1s
        local_dict["recall"] = recalls
        local_dict["precision"] = precisions
        local_dict["inference"] = inferences
        local_dict["retention"] = retentions

        dictionary[dataset][method] = {
            "exp": local_dict, # COULDD
            "original": { # original model
                "acc": accuracy_score(y_true=pred_df["expected_label"], y_pred=pred_df["original_preds_binary"]),
                "f1": f1_score(y_true=pred_df["expected_label"], y_pred=pred_df["original_preds_binary"]),
                "recall": recall_score(y_true=pred_df["expected_label"], y_pred=pred_df["original_preds_binary"]),
                "precision": precision_score(y_true=pred_df["expected_label"], y_pred=pred_df["original_preds_binary"]),
                "inference": compute_change_score(torch.tensor(pred_df["expected_label"].values),
                                                  torch.tensor(pred_df["original_preds_binary"].values),
                                                  torch.tensor(pred_df["original_label"].values)),
                "retention": compute_unchanged_score(torch.tensor(pred_df["expected_label"].values),
                                                     torch.tensor(pred_df["original_preds_binary"].values),
                                                     torch.tensor(pred_df["original_label"].values)),
            }
        }
    pred_df = pd.read_csv(os.path.join(path_to_codex_folder, "gpt_predictions", f"test_predictions_binary_chatgpt_{dataset}.csv"))

    if rm_dup is True:
        pred_df = remove_duplicates(pred_df)

    dictionary[dataset]["chatgpt"] = {
        "acc": accuracy_score(y_true=pred_df["expected_label"], y_pred=pred_df["chatgpt_preds_binary"]),
        "f1": f1_score(y_true=pred_df["expected_label"], y_pred=pred_df["chatgpt_preds_binary"]),
        "recall": recall_score(y_true=pred_df["expected_label"], y_pred=pred_df["chatgpt_preds_binary"]),
        "precision": precision_score(y_true=pred_df["expected_label"], y_pred=pred_df["chatgpt_preds_binary"]),
        "inference": compute_change_score(torch.tensor(pred_df["expected_label"].values),
                                          torch.tensor(pred_df["chatgpt_preds_binary"].values),
                                          torch.tensor(pred_df["original_label"].values)),
        "retention": compute_unchanged_score(torch.tensor(pred_df["expected_label"].values),
                                             torch.tensor(pred_df["chatgpt_preds_binary"].values),
                                             torch.tensor(pred_df["original_label"].values)),
    }

metrics = {
    "f1": "F1",
    "inference": "Changed",
    "retention": "Unchanged"
}

f1prr_df = generate_table(dictionary, metrics)

latex_table = f1prr_df.to_latex(escape=False, index=False, multicolumn_format="c", float_format="{:.2f}".format)
latex_table = '\n'.join(
    [f"{line} \midrule" if "COULDD" in line else line for line in latex_table.split('\n')])
print(latex_table)
