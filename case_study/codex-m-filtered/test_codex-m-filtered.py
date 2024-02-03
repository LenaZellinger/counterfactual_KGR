########################################################################################################################
# This file assesses the performance of pre-trained CoDEx models on the CoDEx-M test set filtered based on Amie3 rules
########################################################################################################################

import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score

from src.codex_functions import get_X_y
from src.eval import tune_thresholds, get_binary_pred
from src.utils import load_model, load_data, load_rules_amie

size = 'm'
models = ['rescal', 'transe', 'complex', 'conve', 'tucker']
model_names = ['RESCAL', 'TransE', 'ComplEx', 'ConvE', 'TuckER']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default='cuda', type=str)
    args = parser.parse_args()

    # load original codex data
    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../../.."))
    data, symmetric_rels, entity_ids, relation_ids = load_data(path_to_codex_folder, size=size)

    # load rules
    rules_frame = load_rules_amie(path_to_codex_folder, size=size, return_conf=True)

    # # create verbalized rule set
    # rules_frame_decoded = copy.deepcopy(rules_frame)
    # rules_frame_decoded['head'] = rules_frame_decoded['head'].apply(lambda x: data.relation_label(x))
    # rules_frame_decoded['antecedent_1'] = rules_frame_decoded['antecedent_1'].apply(lambda x: data.relation_label(x))
    # rules_frame_decoded['antecedent_2'] = rules_frame_decoded['antecedent_2'].apply(lambda x: data.relation_label(x))

    # load inferential benchmark
    filtered_test_set = pd.read_csv(os.path.join(path_to_codex_folder, 'case_study', 'codex-m-filtered', 'codex-m-filtered.csv'), index_col=0)
    filtered_test_set['rules_matched'] = filtered_test_set['rules_matched'].apply(ast.literal_eval)
    filtered_test_set.replace({'head': entity_ids, 'tail': entity_ids, 'relation': relation_ids}, inplace=True) # translate entries to LibKGE ids
    all_rules = np.unique([item for sublist in filtered_test_set['rules_matched'] for item in sublist]) # all rules occuring in the filtered test set

    micro_recalls = []
    macro_recalls = []
    rule_performances = []

    for m in models:
        # load the pre-trained model checkpoints
        model_orig, config, dataset = load_model(path_to_codex_folder, m, size, args.device)
        model_orig.eval()

        # tune the thresholds on the CoDEx validation set
        thresholds = tune_thresholds(model_orig, size, dataset, args.device)

        # compute overall performance on filtered test set
        spo = filtered_test_set[['head', 'relation', 'tail']]
        spo = torch.tensor(spo.values)
        with torch.no_grad():
            X, y = get_X_y(model_orig, spo.to(args.device))
        binary_preds_all = get_binary_pred(X, spo, args.device, thresholds)
        micro_recall = recall_score(y.cpu(), binary_preds_all.cpu()) # since only positive ground-truth labels, accuracy = recall

        # compute rule-wise performance
        recall_per_rule = []
        pca_conf = []
        support = []
        text_description = []
        n_in_test = []

        for r in all_rules:
            # filter data frame according to rule and ensure that rule has at least 5 matches
            filtered_df = filtered_test_set[filtered_test_set['rules_matched'].apply(lambda x: r in x)]
            if filtered_df.shape[0] < 5:
                continue

            # add additional info about the rule
            pca_conf.append(rules_frame.iloc[r, :]['PCA Confidence'])
            support.append(rules_frame.iloc[r, :]['Positive Examples'])
            text_description.append("(" + rules_frame.iloc[r, :]['antecedent_1'] + ", " + rules_frame.iloc[r, :]['antecedent_2'] + ", " + rules_frame.iloc[r, :]['head'] + ")")
            n_in_test.append(filtered_df.shape[0])

            # compute performance solely for triples inferred by this rule
            spo = filtered_df[['head', 'relation', 'tail']]
            spo = torch.tensor(spo.values)
            with torch.no_grad():
                X, y = get_X_y(model_orig, spo.to(args.device))
            binary_preds = get_binary_pred(X, spo, args.device, thresholds)
            recall = recall_score(y.cpu(), binary_preds.cpu())
            recall_per_rule.append(recall)

        macro_recall = np.mean(recall_per_rule) # average performance across rules (some instances might be counted double)

        # append overall and rule-wise recalls
        micro_recalls.append(micro_recall*100)
        macro_recalls.append(macro_recall*100)

        # append individual rule recalls for appendix table
        rule_performances.append(recall_per_rule)

    # main results
    recall_frame = pd.DataFrame({'Overall': micro_recalls, 'Rule-wise': macro_recalls})
    recall_frame.index = model_names
    print(recall_frame.to_latex(float_format="{:.2f}".format))

    # rule-wise performance (appendix)
    all_cols = {'Rule': text_description,
                'Support': support,
                'PCA Confidence': pca_conf,
                '# Test': n_in_test,
                'RESCAL': rule_performances[0],
                'TransE': rule_performances[1],
                'ComplEx': rule_performances[2],
                'ConvE': rule_performances[3],
                'TuckER': rule_performances[4]
        }

    rule_wise_frame = pd.DataFrame(all_cols)
    print(rule_wise_frame.to_latex(float_format="{:.3f}".format, index=False))
