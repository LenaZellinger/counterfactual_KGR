import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


def generate_type_table(dictionary, type_dict, roundeds=2):
    df = []

    indices = []
    for dataset, dataset_name in datasets.items():
        for method, method_name in methods.items():

            df_row_1, df_row_2 = [], []

            # df_without_str_row_1 = [method_name]
            # df_without_str_row_2 = [f"COULDD-{method_name}"]
            for type, type_name in type_dict.items():

                if method != "chatgpt":
                    mean_1 = round(dictionary[dataset][method]["original"][type] * 100, roundeds)

                mean = round(np.mean(dictionary[dataset][method]["exp"][type]) * 100, roundeds)
                std = round(np.std(dictionary[dataset][method]["exp"][type]) * 100, roundeds)

                if method != "chatgpt":
                    if mean - mean_1 > 0:
                        df_row_1.append(mean_1)
                        df_row_2.append(f"\\textbf{{{mean} $\pm$ {std}}}")
                    else:
                        df_row_1.append(f"\\textbf{{{mean_1}}}")
                        df_row_2.append(f"{mean} $\pm$ {std}")
                else:
                    df_row_2.append(mean)

                # df_without_str_row_1.append(mean_1)
                # df_without_str_row_2.append(mean)

            if method != "chatgpt":
                df.append(df_row_1)
            df.append(df_row_2)

            if method != "chatgpt":
                indices.extend([(dataset_name, method_name), (dataset_name, f"COULDD-{method_name}")])
            else:
                indices.extend([(dataset_name, method_name)])

            # df_without_str.append(df_without_str_row_1)
            # df_without_str.append(df_without_str_row_2)

    columns = list(type_dict.values())

    df = pd.DataFrame(df, columns=columns, index=pd.MultiIndex.from_tuples(indices, names=["Dataset", "Method"]))

    return df


methods = {
    "rescal": "RESCAL",
    "transe": "TransE",
    "complex": "ComplEx",
    "conve": "ConvE",
    "tucker": "TuckER",
    "chatgpt": "gpt-3.5-turbo"
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
        if method != "chatgpt":
            pred_df = pd.read_csv(os.path.join(os.path.join(data_dir, f"codex-{dataset}", method, "early_stop_True_batches_uniform", f"test_predictions.csv")))
        else:
            pred_df = pd.read_csv(os.path.join(path_to_codex_folder, "gpt_predictions", f"test_predictions_binary_chatgpt_{dataset}_with_rerun.csv"))

        orig_dict, ccould_dict = {}, {}

        for type, type_df in pred_df.groupby("type"):
            if method != "chatgpt":
                orig_dict[type] = accuracy_score(type_df["expected_label"], type_df["original_preds_binary"])

            if type not in ccould_dict:
                ccould_dict[type] = []
            if method != 'chatgpt':
                ccould_dict[type].extend([accuracy_score(type_df["expected_label"], type_df[f"couldd_preds_{i}_binary"]) for i in
                        range(5)])
            else:
                ccould_dict[type].extend([accuracy_score(type_df["expected_label"], type_df[f"chatgpt_preds_binary"])])


        dictionary[dataset][method] = {
            "exp": ccould_dict,
            "original": orig_dict,
        }

type_dict = {
    'conclusion': "$\\tau^i",
    'far_fact': "$\\tau^f$",
    'near_fact': "$\\tau^n$",
    'head_corr': "$\\tau^i_{h'}$",
    'head_corr_far': "$\\tau^f_{h'}$",
    'head_corr_near': "$\\tau^n_{h'}$",
    'rel_corr': "$\\tau^i_{r'}$",
    'rel_corr_far': "$\\tau^f_{r'}$",
    'rel_corr_near': "$\\tau^n_{r'}$",
    'tail_corr': "$\\tau^i_{t'}$",
    'tail_corr_far': "$\\tau^f_{t'}$",
    'tail_corr_near': "$\\tau^n_{t'}$"
}

type_latex_df = generate_type_table(dictionary, type_dict)

latex_table = type_latex_df.to_latex(escape=False, index=True, multirow=True, multicolumn_format="c", float_format="{:.2f}".format)
print(latex_table)
