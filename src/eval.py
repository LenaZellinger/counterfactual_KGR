"""
This file evaluates the pre-trained CoDEx models and COULDD on our benchmark datasets.
Many of the functions were adapted from https://github.com/tsafavi/codex.
"""

import copy
import gc

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from tqdm import tqdm

from kge.job.train_adapted_couldd import TrainingJob
from kge.model import KgeModel
from src.codex_functions import load_neg_spo, get_X_y, get_threshold, generate_neg_spo


def evaluate_base_model(model: KgeModel,
                        cckg_dataset: pd.DataFrame,
                        dataset,
                        size: str,
                        device: str,
                        eval_codex: bool = False,
                        thresholds: dict = None):
    """
    evaluate the baseline model on CFKGR
    the function assumes that the test cases in the dataset are sorted!
    """

    # compute the prediction for each instance in the CCKG dataset
    model.eval()
    raw_cckg_preds = torch.tensor([]).to(device) # collect raw preds
    binary_cckg_preds = torch.tensor([]).to(device)
    labels_cf = torch.tensor([])
    labels_og = torch.tensor([])

    test_cases = np.unique(cckg_dataset['id'].values)

    for t in test_cases:
        test = cckg_dataset[cckg_dataset['id'] == t]
        test.reset_index(inplace=True, drop=True)  # reset index to start from again (not necessary I think)
        check_test_case(test) # check that test case fulfils all requirements

        # append labels
        raw_preds, binary_preds = score_cckg_instance(test, model, device, thresholds)
        labels_cf = torch.cat([labels_cf, torch.tensor(test['expected_label'].values)])
        labels_og = torch.cat([labels_og, torch.tensor(test['original_label'].values)])

        # append predictions
        raw_cckg_preds = torch.cat([raw_cckg_preds, raw_preds])
        binary_cckg_preds = torch.cat([binary_cckg_preds, binary_preds])

    if eval_codex is True:
        _, _, test_acc_codex, test_f1_codex = evaluate_on_factual(model, size, dataset, device, thresholds)
    else:
        test_acc_codex = np.nan
        test_f1_codex = np.nan

    return raw_cckg_preds.squeeze(1).cpu(), binary_cckg_preds.cpu(), labels_cf.cpu(), labels_og.cpu(), test_acc_codex, test_f1_codex

def apply_COULDD(model_orig: KgeModel,
                    cckg_dataset: pd.DataFrame,
                    dataset,
                    size: str,
                    device: str,
                    config,
                    lr: float,
                    epochs: int,
                    additional_samples: int = 10,
                    additional_sample_type: str = 'uniform',
                    eval_codex: bool = False,
                    thresholds: dict = None,
                    early_stop: bool = False):
    """
    evaluate COULDD on CFKGR
    the function assumes that the test cases in the dataset are sorted!
    """

    # compute the prediction for each instance in the CCKG dataset
    raw_cckg_preds = torch.tensor([]).to(device)
    binary_cckg_preds = torch.tensor([]).to(device)
    labels_cf = torch.tensor([])
    labels_og = torch.tensor([])
    all_test_acc = []
    all_test_f1 = []

    test_cases = np.unique(cckg_dataset['id'].values)

    for t in tqdm(test_cases):
        test = cckg_dataset[cckg_dataset['id'] == t]
        test.reset_index(inplace=True, drop=True)

        # reset model and fine-tune on counterfactual
        model = copy.deepcopy(model_orig)
        train_job = TrainingJob.create(config,
                                       torch.tensor(test['cf_libkge'][0]).to(device),
                                       dataset,
                                       epochs=epochs,
                                       lr=lr,
                                       model=model,
                                       thresholds=thresholds,
                                       additional_samples=additional_samples,
                                       additional_samples_type=additional_sample_type,
                                       early_stop=early_stop)
        train_job.run()

        # clear memory
        if 'cuda' in device:
            torch.cuda.empty_cache()
            gc.collect()

        raw_preds, binary_preds = score_cckg_instance(test, model, device, thresholds)

        if eval_codex is True:
            _, _, test_acc, test_f1 = evaluate_on_factual(model, size, dataset, device, thresholds)
        else:
            test_acc = np.nan
            test_f1 = np.nan

        # append codex performance
        all_test_acc.append(test_acc)
        all_test_f1.append(test_f1)

        # append cckg performance
        raw_cckg_preds = torch.cat([raw_cckg_preds, raw_preds])
        binary_cckg_preds = torch.cat([binary_cckg_preds, binary_preds])

        labels_cf = torch.cat([labels_cf, torch.tensor(test['expected_label'].values)])
        labels_og = torch.cat([labels_og, torch.tensor(test['original_label'].values)])

    # mean performance on codex
    test_acc_codex = np.mean(all_test_acc)
    test_f1_codex = np.mean(all_test_f1)

    return raw_cckg_preds.squeeze(1).cpu(), binary_cckg_preds.cpu(), labels_cf, labels_og, test_acc_codex, test_f1_codex

def score_cckg_instance(test: pd.DataFrame,
                        model: KgeModel,
                        device: str,
                        thresholds: dict):
    """score an instance of the cckg dataset (i.e. all test cases associated with an hypothetical scenario)"""

    model.eval()
    spo = torch.tensor(np.vstack(test['test_inst_libkge'].values))

    with torch.no_grad():
        X, _ = get_X_y(model, spo.to(device))

    binary_preds = get_binary_pred(X, spo, device, thresholds)

    return X, binary_preds

def compute_f1(cckg_labels, cckg_preds):
    f1 = f1_score(cckg_labels.cpu(), cckg_preds.cpu())
    recall = recall_score(cckg_labels.cpu(), cckg_preds.cpu())
    precision = precision_score(cckg_labels.cpu(), cckg_preds.cpu())
    return f1, recall, precision

def compute_unchanged_score(cckg_labels, cckg_preds, original_labels):
    idx = np.where(cckg_labels == original_labels)[0] # labels stay the same
    retained = f1_score(cckg_labels[idx].cpu(), cckg_preds[idx].cpu())
    return retained

def compute_change_score(cckg_labels, cckg_preds, original_labels):
    idx = np.where(cckg_labels != original_labels)[0] # original labels not the same as new ones
    acc = accuracy_score(cckg_labels[idx].cpu(), cckg_preds[idx].cpu())
    return acc

def compute_inference_recall(cckg_labels, cckg_preds, all_types):
    idx = np.where(all_types == 'conclusion')[0]
    recall = recall_score(cckg_labels[idx].cpu(), cckg_preds[idx].cpu())
    return recall

def compute_type_wise_accuracy_changed(cckg_labels, cckg_preds, all_types, original_labels):
    unique_types = np.unique(all_types).tolist()
    acc_dict = dict.fromkeys(unique_types)
    for t in unique_types:
        idx = np.where((all_types == t) & (cckg_labels.cpu().numpy() != original_labels.cpu().numpy()))[0]
        acc = accuracy_score(cckg_labels[idx].cpu(), cckg_preds[idx].cpu())
        acc_dict[t] = acc
    return acc_dict

def compute_type_wise_confusion_matrix(cckg_labels, cckg_preds, all_types):
    unique_types = np.unique(all_types).tolist()
    cf_mat_dict = dict.fromkeys(unique_types)
    for t in unique_types:
        idx = np.where((all_types == t))[0]
        if len(idx) > 0:
            tn, fp, fn, tp = confusion_matrix(cckg_labels[idx].cpu(), cckg_preds[idx].cpu()).ravel()
            cf_mat_dict[t] = (tn, fp, fn, tp)
    return cf_mat_dict

########################################################################################################################
########################################################################################################################
# The following functions were formed based on code in
# https://github.com/tsafavi/codex/blob/master/scripts/tc.py

def get_threshold_global(scores, labels):
    """
    :param scores: torch.tensor of prediction scores
    :param labels: torch.tensor of triple labels
    :return threshold: best decision threshold for these scores
    """

    ###############################################################################
    # Code adapted from https://github.com/tsafavi/codex/blob/master/scripts/tc.py
    # iterative version of global threshold to avoid memory errors
    ###############################################################################

    unique_scores = np.unique(scores).tolist()
    best_acc = 0
    threshold = 0

    for s in unique_scores:
        predictions = (scores >= s).long().t()

        accuracy = (predictions == labels).float().sum(dim=1)
        if accuracy > best_acc: # if better always change threshold
            threshold = s
            best_acc = accuracy
        elif accuracy == best_acc and s < threshold:
            threshold = s

    return threshold

def tune_thresholds(model: KgeModel,
                   size: str,
                   dataset,
                   device: str):
    """tune thresholds on ORIGINAL codex validation set (not CFKGR)"""

    #############################################################################################################
    # Code adapted from https://github.com/tsafavi/codex/blob/master/scripts/tc.py
    # credits to https://github.com/uma-pi1/kge/blob/triple_classification/kge/job/triple_classification.py#L302
    # credits to https://github.com/tsafavi/codex
    #############################################################################################################

    # set model to cpu and eval mode
    model = model.cpu()
    model.eval()

    # load CoDEx valid data and generate or load negatives
    valid_spo = dataset.split('valid').long()
    if size != 'l':
        valid_neg_spo, _ = load_neg_spo(dataset, size=size)
    else:
        valid_neg_spo = generate_neg_spo(dataset, "valid", negative_type="uniform", num_samples=1)

    # score valid triples
    valid_spo_all = torch.cat((valid_spo, valid_neg_spo))
    with torch.no_grad():
        X_valid, y_valid = get_X_y(model, valid_spo, valid_neg_spo)
        y_valid = y_valid.to(device)
    valid_relations = valid_spo_all[:, 1].unique()

    # define dictionary for thresholds
    REL_KEY = -1
    thresholds = {r: -float("inf") for r in valid_relations.tolist()} # corrected from codex code
    thresholds[REL_KEY] = -float("inf")

    # set a threshold for each relation
    for r in valid_relations:
        current_rel = valid_spo_all[:, 1] == r
        threshold = get_threshold(X_valid[current_rel].to(device), y_valid[current_rel].to(device))
        thresholds[r.item()] = threshold

    # tune a global threshold
    thresholds[REL_KEY] = get_threshold_global(X_valid.cpu(), y_valid.cpu())

    # get model back to device
    model.to(device)

    return thresholds

def get_binary_pred(X: torch.tensor,
                    spo_all: torch.tensor,
                    device: str,
                    thresholds: dict):

    ############################################################################
    # Code from https://github.com/tsafavi/codex/blob/master/scripts/tc.py
    # only changes are renaming and relations being a list instead of a tensor
    ############################################################################

    all_preds = torch.zeros(X.shape[0], dtype=torch.long, device=device)
    relations = np.unique(spo_all[:, 1]).tolist()

    for r in relations:  # get predictions based on validation thresholds
        key = r if r in thresholds else -1
        threshold = thresholds[key]

        current_rel = spo_all[:, 1] == r
        predictions = X[current_rel] >= threshold
        all_preds[current_rel] = predictions.view(-1).long()

    return all_preds
########################################################################################################################

### function to evaluate on original codex validation/test sets (not used for results in paper) ########################
def evaluate_on_factual(model: KgeModel,
                        size: str,
                        dataset,
                        device: str,
                        thresholds: dict = None):

    #############################################################################################################
    # credits to: https://github.com/tsafavi/codex/blob/master/scripts/tc.py
    # credits to https://github.com/uma-pi1/kge/blob/triple_classification/kge/job/triple_classification.py#L302 #
    #############################################################################################################

    model.eval()
    valid_spo = dataset.split('valid').long()
    test_spo = dataset.split('test').long()

    if size != 'l':
        valid_neg_spo, test_neg_spo = load_neg_spo(dataset, size=size)
    else:
        valid_neg_spo = generate_neg_spo(dataset, "valid", negative_type="uniform", num_samples=1)
        test_neg_spo = generate_neg_spo(dataset, "test", negative_type="uniform", num_samples=1)

    valid_spo_all = torch.cat((valid_spo, valid_neg_spo))
    test_spo_all = torch.cat((test_spo, test_neg_spo))

    X_valid, y_valid = get_X_y(model, valid_spo.to(device), valid_neg_spo.to(device))
    X_test, y_test = get_X_y(model, test_spo.to(device), test_neg_spo.to(device))

    y_pred_valid = get_binary_pred(X_valid, valid_spo_all, device, thresholds)
    y_pred_test = get_binary_pred(X_test, test_spo_all, device, thresholds)

    y_valid = y_valid.cpu().numpy()
    y_pred_valid = y_pred_valid.cpu().numpy()

    y_test = y_test.cpu().numpy()
    y_pred_test = y_pred_test.cpu().numpy()

    return accuracy_score(y_valid, y_pred_valid), f1_score(y_valid, y_pred_valid), accuracy_score(y_test, y_pred_test), f1_score(y_test, y_pred_test)

########################################################################################################################
################################### end of functions based on the codex code ###########################################

### functions to check data loading and construction ###################################################################
def check_data_load(cckg_labels, original_labels, all_types):
    indices = [i for i, element in enumerate(all_types) if 'corr' in element]

    # check whether the test cases have the assumed cckg labels
    assert all(cckg_labels[np.where(all_types == 'conclusion')[0]] == 1)
    assert all(cckg_labels[np.where(all_types == 'far_fact')[0]] == 1)
    assert all(cckg_labels[np.where(all_types == 'near_fact')[0]] == 1)
    assert all(cckg_labels[indices] == 0)
    assert len(cckg_labels[np.where(all_types == 'conclusion')[0]]) + len(cckg_labels[np.where(all_types == 'far_fact')[0]]) + len(cckg_labels[np.where(all_types == 'near_fact')[0]]) + len(cckg_labels[indices] == 0) == len(cckg_labels)

    # check whether the test cases have the assumed original labels
    assert all(original_labels[np.where(all_types == 'conclusion')[0]] == 0)
    assert all(original_labels[np.where(all_types == 'far_fact')[0]] == 1)
    assert all(original_labels[np.where(all_types == 'near_fact')[0]] == 1)
    assert all(original_labels[indices] == 0)
    assert len(original_labels[np.where(all_types == 'conclusion')[0]]) + len(original_labels[np.where(all_types == 'far_fact')[0]]) + len(original_labels[np.where(all_types == 'near_fact')[0]]) + len(original_labels[indices] == 0) == len(cckg_labels)

def check_test_case(test):
    assert test.shape[0] == 16 # 1 inference, 2 near facts, 1 far fact + 3 corruptions each (4*4 = 16)
    assert test[test['type'] == 'conclusion'].shape[0] == 1
    assert test[test['type'] == 'near_fact'].shape[0] == 2
    assert test[test['type'] == 'far_fact'].shape[0] == 1





