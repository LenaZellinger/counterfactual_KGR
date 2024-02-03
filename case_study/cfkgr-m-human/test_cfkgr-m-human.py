"""
Compute performance on human-annotated data and create table.
"""

import argparse
import ast
import os

import pandas as pd
import torch
import numpy as np
import json

from src.utils import set_seed, load_cfkgr_data_df, remove_duplicates
from src.eval import compute_f1, compute_unchanged_score, compute_change_score, compute_type_wise_confusion_matrix

rm_dup = False

size = 'm'
rule_set = 'automatic'
additional_type = 'uniform'
early_stop = 'True'

min_agreed = 3
labels_to_filter = ['-1.0', 'tie']
label_types = ['automatic', 'human']
model_name = ['rescal', 'transe', 'complex', 'conve', 'tucker']
model_name_table = ['RESCAL', 'COULDD-RESCAL', 'TransE', 'COULDD-TransE', 'ComplEx', 'COULDD-ComplEx', 'ConvE', 'COULDD-ConvE', 'TuckER', 'COULDD-TuckER', 'gpt-3.5-turbo']

f1_list_tab = []
f1_list_human_tab = []
changed_list_tab = []
changed_list_human_tab = []
unchanged_list_tab = []
unchanged_list_human_tab = []
cm_tab = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    # print current setting
    print('Size:', size)
    print('Rule set:', rule_set)
    print('Additional sample type:', additional_type)
    print('Seed:', args.seed)
    print('Early stop:', early_stop)
    print('\n')

    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../../.."))
    for name in model_name:
        # define results folder where model predictions are saved
        valid_folder = os.path.join(path_to_codex_folder, 'results_valid', f'codex-{size}', name, f'early_stop_{early_stop}_batches_{additional_type}')
        test_folder = os.path.join(path_to_codex_folder, 'results_test', f'codex-{size}', name, f'early_stop_{early_stop}_batches_{additional_type}')

        # load classification thresholds
        thresholds = json.load(open(os.path.join(valid_folder, "thresholds.json")))
        thresholds = {int(key): value for key, value in thresholds.items()}

        # load test set and manual annotations
        _, cckg_test_df = load_cfkgr_data_df(path_to_codex_folder, size=size)
        annotations = pd.read_csv(os.path.join(path_to_codex_folder, 'cfkgr_data', 'annotated_data.csv'))

        # preprocess annotations
        if 'og_label' in annotations.columns:
            annotations.rename(columns={'og_label': 'original_label'}, inplace=True)

        # filter annotations (remove uncertain labels, ties, and labels where agreement too small)
        if len(labels_to_filter) > 0:
            annotations = annotations[~annotations['mv_label'].isin(labels_to_filter)].reset_index(drop=True)  # remove unsure and ties
        annotations = annotations[annotations['n_agreed'] >= min_agreed]  # only samples with certain agreement

        # load test predictions
        predictions = pd.read_csv(os.path.join(test_folder, 'test_predictions.csv'), index_col=0)

        # add predictions to frame
        cckg_test_df = predictions.merge(annotations, on=['text', 'head', 'rel', 'tail', 'rule', 'id', 'type', 'expected_label',
                                                            'original_label', 'cf_head', 'cf_rel', 'cf_tail', 'context_head',
                                                            'context_rel', 'context_tail', 'cf_type', 'dataset'], how='left')
        cckg_test_df.dropna(subset='mv_label', inplace=True)
        cckg_test_df.reset_index(inplace=True, drop=True)

        # remove duplicates (different rule but same outcome)
        if rm_dup is True:
            cckg_test_df = remove_duplicates(cckg_test_df)

        # retrieve necessary info from data frame
        cckg_labels_automatic = torch.tensor(cckg_test_df['expected_label'].values)
        cckg_labels_human = torch.tensor(cckg_test_df['mv_label'].values.astype(float).tolist()).long()
        original_labels = torch.tensor(cckg_test_df['original_label'].values)

        all_types = cckg_test_df['type'].values

        cckg_test_df['test_inst_libkge'] = cckg_test_df['test_inst_libkge'].apply(ast.literal_eval)
        preds_orig = torch.tensor(cckg_test_df['original_preds_binary'].values)

        assert np.unique(cckg_labels_human).tolist() == [0, 1] # only 0 and 1
        assert len(cckg_labels_automatic) == len(cckg_labels_human) == len(original_labels) == len(all_types) == cckg_test_df['test_inst_libkge'].shape[0] == len(preds_orig)

        # compute metrics: first with automatic labels, then with mv labels
        # original predictions
        for i, l in enumerate([cckg_labels_automatic, cckg_labels_human]):
            f1 = compute_f1(l, preds_orig)[0]
            changed = compute_change_score(l, preds_orig, original_labels)
            unchanged = compute_unchanged_score(l, preds_orig, original_labels)

            if i == 0: # automatic labels
                f1_list_tab.append(round(f1*100, 2))
                changed_list_tab.append(round(changed*100, 2))
                unchanged_list_tab.append(round(unchanged*100, 2))
            else: # manual labels
                f1_list_human_tab.append(round(f1*100, 2))
                changed_list_human_tab.append(round(changed*100, 2))
                unchanged_list_human_tab.append(round(unchanged*100, 2))

        # confusion matrix for original models (not used in paper)
        conf_mat_dict = compute_type_wise_confusion_matrix(cckg_labels_human, preds_orig, all_types)

        # compute metrics: first with automatic labels, then with mv labels
        # COULDD predictions
        for j, l in enumerate([cckg_labels_automatic, cckg_labels_human]):
            # keep track of results for every run
            f1_list = []
            changed_list = []
            retained_list = []
            conf_mat_by_type = []

            # go through each COULDD run
            for i in range(5):
                preds = torch.tensor(cckg_test_df[f'couldd_preds_{i}_binary'].values)
                f1_list.append(compute_f1(l, preds)[0])
                changed_list.append(compute_change_score(l, preds, original_labels))
                retained_list.append(compute_unchanged_score(l, preds, original_labels))
                conf_mat_by_type.append(compute_type_wise_confusion_matrix(l, preds, all_types))

            f1 = f"{np.round(np.mean(f1_list)*100, 2)} $\pm$ {np.round(np.std(f1_list)*100, 2)}"
            changed = f"{np.round(np.mean(changed_list)*100, 2)} $\pm$ {np.round(np.std(changed_list)*100, 2)}"
            unchanged = f"{np.round(np.mean(retained_list)*100, 2)} $\pm$ {np.round(np.std(retained_list)*100, 2)}"

            if j == 0:
                f1_list_tab.append(f1)
                changed_list_tab.append(changed)
                unchanged_list_tab.append(unchanged)
            else:
                f1_list_human_tab.append(f1)
                changed_list_human_tab.append(changed)
                unchanged_list_human_tab.append(unchanged)

                # build confusion matrix
                confusion_matrix_couldd = {}
                for d in conf_mat_by_type:  # contains list of all confusion matrices across the five runs
                    for key, value in d.items():  # aggregate the confusion matrices per type
                        if key in confusion_matrix_couldd:
                            confusion_matrix_couldd[key].append(value)
                        else:
                            confusion_matrix_couldd[key] = [value]

        cm_row = []
        for key, value_list in confusion_matrix_couldd.items():
            component_means = tuple(sum(x) / len(value_list) for x in zip(*value_list))
            cm_row.extend(component_means)
        cm_tab.append(cm_row)

    # add gpt results
    gpt_responses = pd.read_csv(
            os.path.join(path_to_codex_folder, 'gpt_predictions', 'test_predictions_binary_chatgpt_m_with_rerun.csv'))
    cckg_test_df = cckg_test_df.merge(gpt_responses, on=['text', 'head', 'rel', 'tail', 'rule', 'id', 'type', 'expected_label',
                                              'original_label', 'cf_head', 'cf_rel', 'cf_tail', 'context_head',
                                              'context_rel', 'context_tail', 'cf_type', 'dataset'], how='left')

    gpt_preds = torch.tensor(cckg_test_df['chatgpt_preds_binary'].values)
    cckg_test_df.to_csv(os.path.join(path_to_codex_folder, 'gpt_predictions', 'test_predictions_binary_chatgpt_m_manual.csv'))

    for i, l in enumerate([cckg_labels_automatic, cckg_labels_human]):
        f1 = compute_f1(l, gpt_preds)[0]
        changed = compute_change_score(l, gpt_preds, original_labels)
        unchanged = compute_unchanged_score(l, gpt_preds, original_labels)

        if i == 0:
            f1_list_tab.append(np.round(f1, 4)*100)
            changed_list_tab.append(np.round(changed, 4)*100)
            unchanged_list_tab.append(np.round(unchanged, 4)*100)
        else:
            f1_list_human_tab.append(np.round(f1, 4)*100)
            changed_list_human_tab.append(np.round(changed, 4)*100)
            unchanged_list_human_tab.append(np.round(unchanged, 4)*100)

    conf_mat_gpt = compute_type_wise_confusion_matrix(cckg_labels_human, gpt_preds, all_types)
    gpt_conf_mat = []
    for k in conf_mat_gpt.keys():
        gpt_conf_mat.extend(conf_mat_gpt[k])
    cm_tab.append(gpt_conf_mat)

    #### tables
    # label dist of human labels
    print('Number of hypothetical scenarios:', len(np.unique(cckg_test_df['id'])))
    print('Number of test cases:', cckg_test_df.shape[0])
    print('Number of rules:', len(np.unique(cckg_test_df['rule'])), "\n")

    rows = ['conclusion', 'far_fact', 'near_fact', 'rel_corr']
    assigned_1 = []
    assigned_0 = []
    assigned_unsure = []
    assigned_tie = []
    for r in rows:
        assigned_1.append(cckg_test_df[(cckg_test_df['type'] == r) & (cckg_test_df['mv_label'].values.astype(float) == 1)].shape[0])
        assigned_0.append(cckg_test_df[(cckg_test_df['type'] == r) & (cckg_test_df['mv_label'].values.astype(float) == 0)].shape[0])

    results_df = pd.DataFrame(
        {
         '0': assigned_0,
         '1': assigned_1,
         }
    )
    results_df.index = rows
    print(results_df.to_latex())
    print('\n\n')

    # results table
    results_df = pd.DataFrame(
        {'F1 (E)': f1_list_tab,
         'F1 (H)': f1_list_human_tab,
         'Changed (E)': changed_list_tab,
         'Changed (H)': changed_list_human_tab,
         'Unchanged (E)': unchanged_list_tab,
         'Unchanged (H)': unchanged_list_human_tab})
    results_df.index = model_name_table
    print(results_df.to_latex(float_format="{:.2f}".format))

    # confusion matrix
    cm_table = pd.DataFrame(cm_tab, columns=["TN", "FP", "FN", "TP",
                                             "TN", "FP", "FN", "TP",
                                             "TN", "FP", "FN", "TP",
                                             "TN", "FP", "FN", "TP"])
    cm_table.index = ['COULDD-RESCAL', 'COULDD-TransE', 'COULDD-ComplEx', 'COULDD-ConvE', 'COULDD-TuckER', 'gpt-3.5-turbo']
    print(cm_table.to_latex(float_format="{:.1f}".format))

    # print(f"Number of conclusions:", cckg_test_df[cckg_test_df['type'] == 'conclusion'].shape[0])
    # print(f"Number of far facts:", cckg_test_df[cckg_test_df['type'] == 'far_fact'].shape[0])
    # print(f"Number of near facts:", cckg_test_df[cckg_test_df['type'] == 'near_fact'].shape[0])
    # print(f"Number of rel. corruptions:", cckg_test_df[cckg_test_df['type'] == 'rel_corr'].shape[0])
    # print('\n')
    # print(f"Number of conclusions (label 0):", cckg_test_df[(cckg_test_df['type'] == 'conclusion') & (cckg_test_df['mv_label'].values.astype(float) == 0)].shape[0])
    # print(f"Number of far facts (label 0):", cckg_test_df[(cckg_test_df['type'] == 'far_fact') & (cckg_test_df['mv_label'].values.astype(float) == 0)].shape[0])
    # print(f"Number of near facts (label 0):", cckg_test_df[(cckg_test_df['type'] == 'near_fact') & (cckg_test_df['mv_label'].values.astype(float) == 0)].shape[0])
    # print(f"Number of rel. corruptions (label 0):", cckg_test_df[(cckg_test_df['type'] == 'rel_corr') & (cckg_test_df['mv_label'].values.astype(float) == 0)].shape[0])
    # print('\n')
    # print(f"Number of conclusions (label 1):", cckg_test_df[(cckg_test_df['type'] == 'conclusion') & (cckg_test_df['mv_label'].values.astype(float) == 1)].shape[0])
    # print(f"Number of far facts (label 1):", cckg_test_df[(cckg_test_df['type'] == 'far_fact') & (cckg_test_df['mv_label'].values.astype(float) == 1)].shape[0])
    # print(f"Number of near facts (label 1):", cckg_test_df[(cckg_test_df['type'] == 'near_fact') & (cckg_test_df['mv_label'].values.astype(float) == 1)].shape[0])
    # print(f"Number of rel. corruptions (label 1):", cckg_test_df[(cckg_test_df['type'] == 'rel_corr') & (cckg_test_df['mv_label'].values.astype(float) == 1)].shape[0], "\n")

