import argparse
import copy
import os

import numpy as np
import pandas as pd

from src.utils import load_data, set_seed, load_rules_amie

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='s', type=str, choices=['s', 'm', 'l'], help="size of the CoDEx data")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save', default='False', type=str)
    args = parser.parse_args()

    # if given, set seed
    if args.seed is not None:
        set_seed(args.seed)

    # print current setting
    print('Size:', args.size)
    print('Seed:', args.seed)
    print('\n')

    # load data
    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../.."))
    data, symmetric_rels, entity_ids, relation_ids = load_data(path_to_codex_folder, size=args.size)

    # define storing location
    storing_location_data = os.path.join(path_to_codex_folder, 'cfkgr_data')

    # load data
    full_data = pd.read_json(os.path.join(storing_location_data, f'cfkgr_{args.size}.json'))
    full_data = pd.DataFrame.from_dict(full_data)

    # make sure all information is in df and uniform across dfs
    full_data['cf'] = full_data.apply(lambda row: [row['cf_head'], row['cf_rel'], row['cf_tail']], axis=1)
    full_data['context'] = full_data.apply(lambda row: [row['context_head'], row['context_rel'], row['context_tail']], axis=1)
    full_data['test_inst'] = full_data.apply(lambda row: [row['head'], row['rel'], row['tail']], axis=1)

    if args.size == 'm' and 'og_label' in full_data.columns:
        full_data.rename(columns={'og_label': 'original_label'}, inplace=True)

    if 'cf_head_libkge' not in full_data.columns:
        full_data['cf_head_libkge'] = full_data['cf_head'].map(entity_ids)
    if 'cf_tail_libkge' not in full_data.columns:
        full_data['cf_tail_libkge'] = full_data['cf_tail'].map(entity_ids)
    if 'cf_rel_libkge' not in full_data.columns:
        full_data['cf_rel_libkge'] = full_data['cf_rel'].map(relation_ids)

    if 'context_head_libkge' not in full_data.columns:
        full_data['context_head_libkge'] = full_data['context_head'].map(entity_ids)
    if 'context_tail_libkge' not in full_data.columns:
        full_data['context_tail_libkge'] = full_data['context_tail'].map(entity_ids)
    if 'context_rel_libkge' not in full_data.columns:
        full_data['context_rel_libkge'] = full_data['context_rel'].map(relation_ids)

    if 'head_libkge' not in full_data.columns:
        full_data['head_libkge'] = full_data['head'].map(entity_ids)
    if 'tail_libkge' not in full_data.columns:
        full_data['tail_libkge'] = full_data['tail'].map(entity_ids)
    if 'rel_libkge' not in full_data.columns:
        full_data['rel_libkge'] = full_data['rel'].map(relation_ids)

    full_data['cf_libkge'] = full_data.apply(lambda row: [row['cf_head_libkge'], row['cf_rel_libkge'], row['cf_tail_libkge']], axis=1)
    full_data['context_libkge'] = full_data.apply(lambda row: [row['context_head_libkge'], row['context_rel_libkge'], row['context_tail_libkge']], axis=1)
    full_data['test_inst_libkge'] = full_data.apply(lambda row: [row['head_libkge'], row['rel_libkge'], row['tail_libkge']], axis=1)

    n_rules = len(np.unique(full_data['rule']))
    print('Total number of rules:', n_rules)

    ### handle duplicated hypothetical scenarios (make sure they are either both in valid or both in test); note that they still have different conclusions
    detect_dup = full_data[full_data['type'] == 'conclusion']
    duplicated_entries = detect_dup[detect_dup.duplicated(subset='cf')]['id'].tolist()
    duplicate_cf = np.unique(full_data[full_data['id'].isin(duplicated_entries)]['cf']).tolist()
    rules_with_dups = []
    for c in duplicate_cf:
        # find duplicates
        dups = np.unique(full_data[full_data['cf'].isin([c])]['id'].values).tolist()
        # find which rule they belong to
        rules = np.unique(full_data[full_data['id'].isin(dups)]['rule'].values).tolist()
        rules_with_dups.append(rules)

    # get rule combs which cannot occur together
    all_rules = np.unique(full_data['rule']).tolist()
    val_rules = []
    test_rules = []

    if len(rules_with_dups) > 0:
        unique_sublists = set()
        for sublist in rules_with_dups:
            sublist_tuple = tuple(sublist)
            unique_sublists.add(sublist_tuple)
        for r in unique_sublists:
            selection = np.random.choice(2) # valid or test
            if selection == 0 and len(val_rules) + len(r) <= 5:
                val_rules.extend(sublist)
            # remove from assignable to validation (either was already added or should be in test)
            for e in r:
                all_rules.remove(e) # don't sample this again
    full_data.reset_index(inplace=False, drop=True)

    # sample 5 rules for validation
    if len(val_rules) < 5:
        rules = np.random.choice(all_rules, 5-len(val_rules), replace=False)
        val_rules.extend(rules.tolist())
    remaining_rules = np.unique(full_data['rule']).tolist()
    test_rules = copy.deepcopy(remaining_rules)

    for r in val_rules:
        test_rules.remove(r)

    print('Number of validation rules:', len(val_rules))
    print('Number of test rules:', len(test_rules), "\n")
    assert len(set(val_rules) & set(test_rules)) == 0
    assert n_rules == len(val_rules) + len(test_rules)
    assert len(val_rules) == 5

    val_frame = full_data[full_data['rule'].isin(val_rules)]
    val_frame.reset_index(inplace=True, drop=True)
    test_frame = full_data[full_data['rule'].isin(test_rules)]
    test_frame.reset_index(inplace=True, drop=True)
    assert val_frame.shape[0] + test_frame.shape[0] == full_data.shape[0]

    # check that really no overlap between val and test counterfactuals
    no_dups = pd.concat([val_frame[val_frame['type'] == 'conclusion'].drop_duplicates(subset='cf'), test_frame[test_frame['type'] == 'conclusion'].drop_duplicates(subset='cf')], ignore_index=True)
    dup = no_dups[no_dups.duplicated(subset='cf', keep=False)]
    assert no_dups.shape[0] == no_dups.drop_duplicates(subset='cf').shape[0]

    print('Number of hypothetical scenarios (validation):', len(np.unique(val_frame['id'].values)))
    print('Number of hypothetical scenarios (test):', len(np.unique(test_frame['id'].values)))
    print('Total number of instances (validation):', val_frame.shape[0])
    print('Total number of instances (test):', test_frame.shape[0])

    # load rules
    rules = load_rules_amie(path_to_codex_folder, size=args.size)
    val_rules_frame = rules.iloc[val_rules, :]
    test_rules_frame = rules.iloc[test_rules, :]

    ### last duplicate check
    test_cf = copy.deepcopy(test_frame[['cf_head', 'cf_rel', 'cf_tail', 'head', 'rel', 'tail']])
    test_cf = test_cf[test_cf.duplicated(keep=False)]
    non_unique = test_frame.iloc[test_cf.index, :]
    non_unique = non_unique[non_unique['type'] == 'conclusion']
    print('Conclusions total:', test_frame[test_frame['type'] == 'conclusion'].shape[0])
    print('Non-unique:', non_unique.shape[0])
    print('Conclusions that maybe should be filtered out:', non_unique.shape[0] // 2)

    ### save the dataframes and rules
    if args.save == 'True':
        val_frame.to_csv(os.path.join(storing_location_data, f'cfkgr_{args.size}_val.csv'), index=False)
        test_frame.to_csv(os.path.join(storing_location_data, f'cfkgr_{args.size}_test.csv'), index=False)

        val_rules_frame.to_csv(os.path.join(storing_location_data, f'cfkgr_{args.size}_val_rules.csv'), index=False)
        test_rules_frame.to_csv(os.path.join(storing_location_data, f'cfkgr_{args.size}_test_rules.csv'), index=False)