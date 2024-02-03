"""
This file tunes the COULDD hyperparameters on the CFKGR validation set.
Note that the code assumes that the dataframes are ordered by test instance (column 'id' in our datasets)
"""

import argparse
import copy
import json
import os
import time

import joblib
import numpy as np
import optuna
import torch

from src.eval import evaluate_base_model, apply_COULDD, tune_thresholds, compute_f1, compute_unchanged_score, \
    compute_change_score, check_data_load
from src.utils import set_seed, load_model, load_cfkgr_data_df

epochs_range = [1, 3, 5, 7, 9, 11, 13, 15, 20]  # epochs range when no early stopping is used (not used for paper)
additional_samples_range = [0, 127, 255, 511, 1023]
lr_range = [0.001, 0.01, 0.1, 0.15, 0.2]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="s", type=str, choices=['s', 'm', 'l'])
    parser.add_argument("--seed", default=0, type=int)
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

    # 0 additional samples not compatible with conve
    if args.model_name == 'conve':
        additional_samples_range.remove(0)  # cannot have 0 additional samples for ConvE because of BatchNorm

    # early stop setting; always True for paper
    if args.early_stop == 'False':
        early_stop = False
    elif args.early_stop == 'True':
        early_stop = True
    else:
        raise ValueError('Invalid early stop argument.')

    # define results folder
    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../.."))
    results_folder = os.path.join(path_to_codex_folder, 'results_valid', f'codex-{args.size}', args.model_name, f'early_stop_{args.early_stop}_batches_{args.additional_type}')
    os.makedirs(results_folder, exist_ok=True)

    # load validation set
    cckg_valid_df, _ = load_cfkgr_data_df(path_to_codex_folder, size=args.size)
    print('Number of validation instances:', len(np.unique(cckg_valid_df['id'])))
    print('Number of validation cases:', cckg_valid_df.shape[0], "\n")

    # load the pre-trained model checkpoints and tune threshold
    model_orig, config, dataset = load_model(path_to_codex_folder, args.model_name, args.size, args.device)
    thresholds = tune_thresholds(model_orig, args.size, dataset, args.device)

    # retrieve expected label, original label, and type assignment of data points
    cckg_labels = torch.tensor(cckg_valid_df['expected_label'].values)
    original_labels = torch.tensor(cckg_valid_df['original_label'].values)
    all_types = cckg_valid_df['type'].values
    check_data_load(cckg_labels, original_labels, all_types) # check that labels are assigned as expected

    cckg_preds_valid_orig, binary_preds_orig, cf_labels_orig, og_labels_orig, _, _ = evaluate_base_model(model_orig, cckg_valid_df, dataset, thresholds=thresholds, size=args.size, device=args.device, eval_codex=False)

    # add preds to dataframe
    cckg_valid_with_preds = copy.deepcopy(cckg_valid_df)
    cckg_valid_with_preds['original_preds'] = cckg_preds_valid_orig
    cckg_valid_with_preds['original_preds_binary'] = binary_preds_orig.cpu().numpy()

    # compute scores
    acc_inferences_orig = compute_change_score(cf_labels_orig, binary_preds_orig, og_labels_orig)
    retention_orig = compute_unchanged_score(cf_labels_orig, binary_preds_orig, og_labels_orig)
    f1_orig, recall_orig, precision_orig = compute_f1(cf_labels_orig, binary_preds_orig)
    print(f'Original F1: {f1_orig}, Precision: {precision_orig}, Recall: {recall_orig}')
    print('Original Acc. (changes):', acc_inferences_orig)
    print('Original F1 (retained):', retention_orig)

    ############################################################################################
    # choose hyperparameters on validation set
    ############################################################################################

    # adjust config to fit desired training style
    config.set('train.type', 'couldd')
    if config.get('train.type') == 'couldd':
        config.set('negative_sampling.num_samples.o', 50)
        config.set('negative_sampling.num_samples.s', 50)
        config.set('negative_sampling.num_samples.p', 0)
        config.set('negative_sampling.shared', False)
        config.set('negative_sampling.filtering.o', False)
        config.set('negative_sampling.filtering.s', False)
        config.set('negative_sampling.implementation', 'batch')

    print('Number of negatives (head):', config.get('negative_sampling.num_samples.s'))
    print('Number of negatives (tail):', config.get('negative_sampling.num_samples.o'))
    print('Shared sampling:', config.get("negative_sampling.shared"))
    print('Replacement:', config.get("negative_sampling.with_replacement"), "\n")

    def objective(trial):
        if args.early_stop == 'True':
            params = {
                'lr': trial.suggest_categorical('lr', lr_range),
                'samples': trial.suggest_categorical('samples', additional_samples_range),
            }
            epochs = 20
        else:
            params = {
                'lr': trial.suggest_categorical('lr', lr_range),
                'samples': trial.suggest_categorical('samples', additional_samples_range),
                'epochs': trial.suggest_categorical('epochs', epochs_range),
            }
            epochs = params['epochs']

        start = time.time()
        print(f'Current params:', params)

        preds, binary_preds, cf_labels_couldd, og_labels_couldd, _, _ = apply_COULDD(model_orig,
                                                                    cckg_valid_df,
                                                                    dataset,
                                                                    args.size,
                                                                    args.device,
                                                                    config,
                                                                    epochs=epochs,
                                                                    lr=params['lr'],
                                                                    eval_codex=False,
                                                                    additional_samples=params['samples'],
                                                                    thresholds=thresholds,
                                                                    early_stop=early_stop,
                                                                    additional_sample_type=args.additional_type)

        cckg_valid_with_preds[f'couldd_preds_trial_{trial.number}'] = preds
        assert all(cf_labels_couldd == cf_labels_orig)
        assert all(og_labels_couldd == og_labels_orig)

        # compute binary preds
        cckg_valid_with_preds[f'couldd_preds_trial_{trial.number}_binary'] = binary_preds.cpu().numpy()

        # compute scores
        inferences = compute_change_score(cckg_labels, binary_preds, original_labels)
        unchanged = compute_unchanged_score(cckg_labels, binary_preds, original_labels)
        f1, recall, precision = compute_f1(cckg_labels, binary_preds)
        end = time.time()

        # print performance of current combination
        print('Total time for one combination:', end-start)
        print('F1:', f1)
        print('Acc. (changes):', inferences)
        print('F1 (unchanged):', unchanged)
        return f1

    # perform the optimization
    start = time.time()
    if early_stop is True:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.GridSampler(seed=args.seed, search_space={'lr': lr_range, 'samples': additional_samples_range}))
        study.optimize(objective)
    else:
        # RandomSampler if no early stop is used (not used in paper)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=args.seed))
        study.optimize(objective, n_trials=20)
    end = time.time()
    print('Total time spent on optimization:', end-start)
    best_trial = study.best_trial.number
    cckg_valid_with_preds['best_couldd_preds'] = cckg_valid_with_preds[f'couldd_preds_trial_{best_trial}']

    if args.save == 'True':
        joblib.dump(study, os.path.join(results_folder, "optuna_study_couldd.pkl"))
        json.dump(thresholds, open(os.path.join(results_folder, "thresholds.json"), 'w'))
        cckg_valid_with_preds.to_csv(os.path.join(results_folder, 'valid_predictions.csv'))