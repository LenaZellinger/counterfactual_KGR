import ast
import copy
import os
import random
import re

import numpy as np
import pandas as pd
import torch

from codex.codex import Codex
from kge.config import Config
from kge.dataset import Dataset
from kge.model import KgeModel
from kge.util.io import load_checkpoint

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_data(path_to_codex_folder: str, size: str):
    """Load CoDEx data + necessary information"""
    data = Codex(size=size)

    # Safavi and Koutra 2020
    symmetric_rels = ['P530', 'P47', 'P3373', 'P26', 'P451'] # diplomatic relation, shares border with, sibling, spouse, unmarried partner

    # load entity and relation ids for LibKGE
    entity_ids = pd.read_csv(os.path.join(path_to_codex_folder, "kge", "data", "codex-" + size, "entity_ids.del"),
                                sep="\t", header=None, names=["id", "entity"])
    entity_ids = dict(zip(entity_ids["entity"], entity_ids["id"]))

    relation_ids = pd.read_csv(os.path.join(path_to_codex_folder, "kge", "data", "codex-" + size, "relation_ids.del"),
                                sep="\t", header=None, names=["id", "relation"])
    relation_ids = dict(zip(relation_ids["relation"], relation_ids["id"]))

    return data, symmetric_rels, entity_ids, relation_ids

def load_rules_amie(path_to_codex_folder: str, size: str, min_conf: float = 0, min_pos: int = 1, return_conf: bool = False):
    """
    load the manually annotated rules
    :param path_to_codex_folder: path to the CoDEx folder
    :param size: CoDEx subset size
    :param min_conf: minimum required PCA confidence
    :param min_pos: minimum number of positive examples of rule
    :return: Amie3 rules
    """

    rules_codex = pd.read_csv(os.path.join(path_to_codex_folder, "analysis", "paths", "codex-" + size, "paths.tsv"), sep="\t")

    if size == 's':
        assert rules_codex.shape[0] == 26

    elif size == 'm':
        assert rules_codex.shape[0] == 44

    elif size == 'l':
        assert rules_codex.shape[0] == 93

    rules_codex = rules_codex[(rules_codex['PCA Confidence'] >= min_conf) & (rules_codex['Positive Examples'] >= min_pos)]
    rules_codex = rules_codex.sort_values(by='PCA Confidence', ascending=False)
    rules_codex = rules_codex[rules_codex['Length'] == 2]
    rules_codex.reset_index(inplace=True, drop=True)
    rules = rules_codex["Rule"]
    rules_parsed = [re.findall(r"P\d+", r) for r in rules]
    rules_frame = pd.DataFrame(columns=["antecedent_1", "antecedent_2", "head"], data=rules_parsed)

    if size == 's':
        assert rules_frame.shape[0] == 18

    elif size == 'm':
        assert rules_frame.shape[0] == 31

    elif size == 'l':
        assert rules_frame.shape[0] == 45

    if return_conf is True:
        rules_frame['PCA Confidence'] = rules_codex['PCA Confidence']
        rules_frame['Std Confidence'] = rules_codex['Std Confidence']
        rules_frame['Positive Examples'] = rules_codex['Positive Examples']

    return rules_frame

def load_model(path_to_codex_folder: str, model_name: str, size: str, device: str):
    storing_loc = os.path.join(path_to_codex_folder, "models", "link-prediction", f"codex-{size}", model_name)
    if model_name in ['transe', 'complex', 'conve', 'rescal', 'tucker']:
        model_checkpoint = os.path.join(storing_loc, "checkpoint_best.pt")
        checkpoint = load_checkpoint(model_checkpoint, device=device)
        model = KgeModel.create_from(checkpoint)
        config = model.config
        dataset = model.dataset

    else:
        model = torch.load(os.path.join(storing_loc, "checkpoint_best.pt"))['model']
        model = model.to(device)
        config = Config(folder=storing_loc)
        config.load(os.path.join(storing_loc, "config.yaml"), create=True)
        dataset_config = Config()
        dataset_config.load(os.path.join(path_to_codex_folder, 'kge', 'data', f'codex-{size}', 'dataset.yaml'))
        dataset = Dataset.create(config=dataset_config)

    return model, config, dataset

def convert_to_libkge_id(triple, entity_ids, relation_ids):
    triple = [entity_ids[triple[0]], relation_ids[triple[1]], entity_ids[triple[2]]]
    return triple

def display_triple(triple, data):
    return data.entity_label(triple[0]), data.relation_label(triple[1]), data.entity_label(triple[2])

def create_libkge_entry(dataset,
                            cf,
                            context,
                            conclusion,
                            corruptions,
                            corruptions_far,
                            facts_to_retain_near,
                            facts_to_retain_far,
                            rule_id,
                            example_id,
                            size,
                            type,
                            entity_ids,
                            relation_ids
                        ):

    """Create a data entry for CFKGR. Corruptions for near facts are not yet included and are added in the file 'create_additional_negatives.py'.'"""

    # convert to Libkge id
    cf_libkge = convert_to_libkge_id(cf, entity_ids, relation_ids)
    context_libkge = convert_to_libkge_id(context, entity_ids, relation_ids)
    conclusion_libkge = convert_to_libkge_id(conclusion, entity_ids, relation_ids)

    facts_to_retain_near_libkge = []
    for f in facts_to_retain_near:
        fact = convert_to_libkge_id(f, entity_ids, relation_ids)
        facts_to_retain_near_libkge.append(fact)

    facts_to_retain_far_libkge = []
    for f in facts_to_retain_far:
        fact = convert_to_libkge_id(f, entity_ids, relation_ids)
        facts_to_retain_far_libkge.append(fact)

    corruptions_libkge = []
    for f in corruptions:
        fact = convert_to_libkge_id(f, entity_ids, relation_ids)
        corruptions_libkge.append(fact)

    corruptions_far_libkge = []
    for f in corruptions_far:
        fact = convert_to_libkge_id(f, entity_ids, relation_ids)
        corruptions_far_libkge.append(fact)

    # add conclusion
    joint_information = {'cf_head': cf[0], 'cf_rel': cf[1], 'cf_tail': cf[2],
                        'cf_head_libkge': cf_libkge[0], 'cf_rel_libkge': cf_libkge[1], 'cf_tail_libkge': cf_libkge[2],
                        'context_head': context[0], 'context_rel': context[1], 'context_tail': context[2],
                        'context_head_libkge': context_libkge[0], 'context_rel_libkge': context_libkge[1], 'context_tail_libkge': context_libkge[2],
                        'rule': rule_id, 'id': example_id,
                        'cf_type': type, 'dataset': f'cfkgr_{size}'}

    conclusion_dict = {'head': conclusion[0], 'rel': conclusion[1], 'tail': conclusion[2],
                        'head_libkge': conclusion_libkge[0], 'rel_libkge': conclusion_libkge[1], 'tail_libkge': conclusion_libkge[2],
                        'type': 'conclusion', 'expected_label': 1, 'original_label': 0}
    conclusion_dict.update(joint_information)
    dataset.append(conclusion_dict)

    # add corruptions
    head_corr_dict = {'head': corruptions[0][0], 'rel': corruptions[0][1], 'tail': corruptions[0][2],
                        'head_libkge': corruptions_libkge[0][0], 'rel_libkge': corruptions_libkge[0][1], 'tail_libkge': corruptions_libkge[0][2],
                        'type': 'head_corr', 'expected_label': 0, 'original_label': 0}
    head_corr_dict.update(joint_information)
    dataset.append(head_corr_dict)

    tail_corr_dict = {'head': corruptions[1][0], 'rel': corruptions[1][1], 'tail': corruptions[1][2],
                        'head_libkge': corruptions_libkge[1][0], 'rel_libkge': corruptions_libkge[1][1], 'tail_libkge': corruptions_libkge[1][2],
                        'type': 'tail_corr', 'expected_label': 0, 'original_label': 0}
    tail_corr_dict.update(joint_information)
    dataset.append(tail_corr_dict)

    rel_corr_dict = {'head': corruptions[2][0], 'rel': corruptions[2][1], 'tail': corruptions[2][2],
                    'head_libkge': corruptions_libkge[2][0], 'rel_libkge': corruptions_libkge[2][1], 'tail_libkge': corruptions_libkge[2][2],
                    'type': 'rel_corr', 'expected_label': 0, 'original_label': 0}
    rel_corr_dict.update(joint_information)
    dataset.append(rel_corr_dict)

    head_corr_far_dict = {'head': corruptions_far[0][0], 'rel': corruptions_far[0][1], 'tail': corruptions_far[0][2],
                        'head_libkge': corruptions_far_libkge[0][0], 'rel_libkge': corruptions_far_libkge[0][1], 'tail_libkge': corruptions_far_libkge[0][2],
                        'type': 'head_corr_far', 'expected_label': 0,  'original_label': 0}
    head_corr_far_dict.update(joint_information)
    dataset.append(head_corr_far_dict)

    tail_corr_far_dict = {'head': corruptions_far[1][0], 'rel': corruptions_far[1][1], 'tail': corruptions_far[1][2],
                        'head_libkge': corruptions_far_libkge[1][0], 'rel_libkge': corruptions_far_libkge[1][1], 'tail_libkge': corruptions_far_libkge[1][2],
                        'type': 'tail_corr_far', 'expected_label': 0, 'original_label': 0}
    tail_corr_far_dict.update(joint_information)
    dataset.append(tail_corr_far_dict)

    rel_corr_far_dict = {'head': corruptions_far[2][0], 'rel': corruptions_far[2][1], 'tail': corruptions_far[2][2],
                        'head_libkge': corruptions_far_libkge[2][0], 'rel_libkge': corruptions_far_libkge[2][1], 'tail_libkge': corruptions_far_libkge[2][2],
                        'type': 'rel_corr_far', 'expected_label': 0, 'original_label': 0}
    rel_corr_far_dict.update(joint_information)
    dataset.append(rel_corr_far_dict)

    # add facts that should be retained
    for i, f in enumerate(facts_to_retain_near):
        fact_near_dict = {'head': facts_to_retain_near[i][0], 'rel': facts_to_retain_near[i][1], 'tail': facts_to_retain_near[i][2],
             'head_libkge': facts_to_retain_near_libkge[i][0], 'rel_libkge': facts_to_retain_near_libkge[i][1], 'tail_libkge': facts_to_retain_near_libkge[i][2],
             'type': 'near_fact', 'expected_label': 1, 'original_label': 1}
        fact_near_dict.update(joint_information)
        dataset.append(fact_near_dict)

    for i, f in enumerate(facts_to_retain_far):
        fact_far_dict = {'head': facts_to_retain_far[i][0], 'rel': facts_to_retain_far[i][1], 'tail': facts_to_retain_far[i][2],
             'head_libkge': facts_to_retain_far_libkge[i][0], 'rel_libkge': facts_to_retain_far_libkge[i][1], 'tail_libkge': facts_to_retain_far_libkge[i][2],
             'type': 'far_fact', 'expected_label': 1, 'original_label': 1}
        fact_far_dict.update(joint_information)
        dataset.append(fact_far_dict)

    return dataset

def load_cfkgr_data_df(path_to_codex_folder, size):
    """
    load cfkgr data
    """

    cfkgr_dataset_valid_df = pd.read_csv(os.path.join(path_to_codex_folder, 'cfkgr_data', f'cckg_{size}_automatic_val.csv'))
    cfkgr_dataset_valid_df['test_inst_libkge'] = cfkgr_dataset_valid_df['test_inst_libkge'].apply(ast.literal_eval)
    cfkgr_dataset_valid_df['cf_libkge'] = cfkgr_dataset_valid_df['cf_libkge'].apply(ast.literal_eval)

    if np.array_equal(cfkgr_dataset_valid_df['id'].values, np.sort(cfkgr_dataset_valid_df['id'].values)) is False:
        cfkgr_dataset_valid_df.sort_values('id', inplace=True) # if dataframe not sorted according to ids; do so

    cfkgr_dataset_test_df = pd.read_csv(os.path.join(path_to_codex_folder, 'cfkgr_data', f'cckg_{size}_automatic_test.csv'))
    cfkgr_dataset_test_df['test_inst_libkge'] = cfkgr_dataset_test_df['test_inst_libkge'].apply(ast.literal_eval)
    cfkgr_dataset_test_df['cf_libkge'] = cfkgr_dataset_test_df['cf_libkge'].apply(ast.literal_eval)

    if np.array_equal(cfkgr_dataset_test_df['id'].values, np.sort(cfkgr_dataset_test_df['id'].values)) is False:
        cfkgr_dataset_test_df.sort_values('id', inplace=True) # if dataframe not sorted according to ids; do so

    return cfkgr_dataset_valid_df, cfkgr_dataset_test_df

def remove_duplicates(test_frame):
    """
    keep only one random duplicated entry
    """
    test_cf = copy.deepcopy(test_frame[['cf_head', 'cf_rel', 'cf_tail', 'head', 'rel', 'tail', 'type']])
    test_cf = test_cf[test_cf['type'] == 'conclusion']
    test_cf = test_cf.sample(frac=1)
    dups = test_cf[test_cf.duplicated(keep='first')].index
    ids_to_remove = test_frame.iloc[dups, :]['id'].values.tolist()
    test_frame_no_dup = test_frame[~test_frame['id'].isin(ids_to_remove)].reset_index(drop=True)
    return test_frame_no_dup
