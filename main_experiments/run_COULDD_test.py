"""
This file evaluates COULDD on the CFKGR test set.
Note that the code assumes that the dataframes are ordered by test instance (column 'id' in our datasets).
"""

import argparse
import copy
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch

from src.eval import evaluate_base_model, apply_COULDD, compute_f1, compute_unchanged_score, compute_change_score
from src.utils import set_seed, load_model, load_cfkgr_data_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="s", type=str, choices=['s', 'm', 'l'])
    parser.add_argument("--seed", default=0, type=int, help="seed 0 used for results in paper")
    parser.add_argument("--model_name", default="rescal", type=str)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--additional_type", default='uniform', type=str, help="for results in paper always 'uniform'")
    parser.add_argument('--early_stop', default='True', type=str, help="for results in paper always 'True'")
    parser.add_argument('--save', default='False', type=str)
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    # print current setting
    print('Model:', args.model_name)
    print('Size:', args.size)
    print('Additional sample type:', args.additional_type)
    print('Save:', args.save)
    print('Seed:', args.seed)
    print('Early stop:', args.early_stop)

    # define results folder
    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../.."))
    valid_folder = os.path.join(path_to_codex_folder, 'results_valid', f'codex-{args.size}', args.model_name, f'early_stop_{args.early_stop}_batches_{args.additional_type}')
    test_folder = os.path.join(path_to_codex_folder, 'results_test', f'codex-{args.size}', args.model_name, f'early_stop_{args.early_stop}_batches_{args.additional_type}')
    os.makedirs(test_folder, exist_ok=True)
    all_results = pd.DataFrame(columns=['model', 'f1', 'f1_std', 'recall', 'recall_std', 'precision', 'precision_std', 'changed', 'changed_std', 'unchanged', 'unchanged_std'])

    _, cckg_test_df = load_cfkgr_data_df(path_to_codex_folder, size=args.size)
    # cckg_test_df = cckg_test_df.sample(frac=1).reset_index(drop=True) # sanity check

    cckg_labels = torch.tensor(cckg_test_df['expected_label'].values)
    original_labels = torch.tensor(cckg_test_df['original_label'].values)
    all_types = cckg_test_df['type'].values
    print('Number of test instances:', len(np.unique(cckg_test_df['id'])))
    print('Number of test cases:', cckg_test_df.shape[0], "\n")

    # load the pre-trained model checkpoint and thresholds
    model_orig, config, dataset = load_model(path_to_codex_folder, args.model_name, args.size, args.device)
    thresholds = json.load(open(os.path.join(valid_folder, "thresholds.json")))
    thresholds = {int(key): value for key, value in thresholds.items()}

    if args.early_stop == 'False':
        early_stop = False
    elif args.early_stop == 'True':
        early_stop = True
    else:
        raise ValueError

    ### evaluate pre-trained model
    cckg_preds_test_orig, binary_preds_orig, cf_labels_orig, og_labels_orig, acc_codex_test, f1_codex_test = evaluate_base_model(model_orig, cckg_test_df, dataset, thresholds=thresholds, size=args.size, device=args.device, eval_codex=False)

    # add preds to dataframe
    cckg_test_with_preds = copy.deepcopy(cckg_test_df)
    cckg_test_with_preds['original_preds'] = cckg_preds_test_orig
    cckg_test_with_preds['original_preds_binary'] = binary_preds_orig.cpu().numpy()
    acc_inferences_orig = compute_change_score(cf_labels_orig, binary_preds_orig, og_labels_orig)
    retention_orig = compute_unchanged_score(cf_labels_orig, binary_preds_orig, og_labels_orig)
    f1_orig, recall_orig, precision_orig = compute_f1(cf_labels_orig, binary_preds_orig)

    print('Original F1:', round(f1_orig, 4)*100)
    print('Original Acc. (changes):', round(acc_inferences_orig, 4)*100)
    print('Original F1 (retained):', round(retention_orig, 4)*100)
    print('Original CoDEx accuracy:', round(acc_codex_test, 4) * 100)
    print('Original CoDEx F1:', round(f1_codex_test, 4) * 100)

    ### evaluate COULDD
    # adjust config to fit desired training style
    config.set('train.type', 'couldd')
    if config.get('train.type') == 'couldd':
        config.set('negative_sampling.num_samples.o', 50)
        config.set('negative_sampling.num_samples.s', 50)
        config.set('negative_sampling.num_samples.p', 0)
        config.set('negative_sampling.shared', False)
        config.set('negative_sampling.filtering.o', False)
        config.set('negative_sampling.filtering.s', False)
        config.set('negative_sampling.implementation', 'triple')

    print('Number of negatives (head):', config.get('negative_sampling.num_samples.s'))
    print('Number of negatives (tail):', config.get('negative_sampling.num_samples.o'))
    print('Shared sampling:', config.get("negative_sampling.shared"))
    print('Replacement:', config.get("negative_sampling.with_replacement"), "\n")

    study = joblib.load(os.path.join(valid_folder, 'optuna_study.pkl'), 'r')
    best_trial = study.best_trial
    best_params = study.best_params
    print('Chosen params:', best_params)
    if 'epochs' in best_params:
        epochs = best_params['epochs']
    else:
        epochs = 20

    f1_list = []
    recall_list = []
    precision_list = []
    acc_inf_list = []
    retention_list = []
    codex_acc_list = []
    codex_f1_list = []

    for i in range(5):
        start = time.time()
        preds, binary_preds, cf_labels_couldd, og_labels_couldd, acc_codex_test, f1_codex_test = apply_COULDD(model_orig,
                                                                                    cckg_test_df,
                                                                                    dataset,
                                                                                    args.size,
                                                                                    args.device,
                                                                                    config,
                                                                                    epochs=epochs,
                                                                                    lr=best_params['lr'],
                                                                                    eval_codex=False,
                                                                                    additional_samples=best_params['samples'],
                                                                                    thresholds=thresholds,
                                                                                    early_stop=early_stop,
                                                                                    additional_sample_type=args.additional_type)
        assert all(cf_labels_couldd == cf_labels_orig)
        assert all(og_labels_couldd == og_labels_orig)

        acc_inferences = compute_change_score(cf_labels_couldd, binary_preds, og_labels_couldd)
        unchanged = compute_unchanged_score(cf_labels_couldd, binary_preds, og_labels_couldd)
        f1, recall, precision = compute_f1(cf_labels_couldd, binary_preds)
        end = time.time()
        print('Total time for one combination:', end-start)

        cckg_test_with_preds[f'couldd_preds_{i}'] = preds
        cckg_test_with_preds[f'couldd_preds_{i}_binary'] = binary_preds.cpu().numpy()

        # main metrics
        f1_list.append(f1)
        acc_inf_list.append(acc_inferences)
        retention_list.append(unchanged)
        codex_acc_list.append(acc_codex_test)
        codex_f1_list.append(f1_codex_test)

        # additional info
        recall_list.append(recall)
        precision_list.append(precision)

    print('\n')
    print(f'Model: {args.model_name}')
    print(f'COULDD Micro F1: {round(np.mean(f1_list)*100, 2)} ({round(np.std(f1_list)*100, 2)})')
    print(f'COULDD Acc. (changes).: {round(np.mean(acc_inf_list)*100, 2)} ({round(np.std(acc_inf_list)*100, 2)})')
    print(f'COULDD F1 (retained): {round(np.mean(retention_list)*100, 2)} ({round(np.std(retention_list)*100, 2)})')
    print(f'COULDD CoDEx Acc.: {round(np.mean(codex_acc_list) * 100, 2)} ({round(np.std(codex_acc_list) * 100, 2)})')
    print(f'COULDD CoDEx F1: {round(np.mean(codex_f1_list) * 100, 2)} ({round(np.std(codex_f1_list) * 100, 2)})')
    print('\n')

    if args.save == 'True':
        cckg_test_with_preds.to_csv(os.path.join(test_folder, 'test_predictions.csv'))
