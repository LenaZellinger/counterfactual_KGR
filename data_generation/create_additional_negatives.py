"""
This file generates additional negatives for the data. In particular, it creates corruptions for near and far facts.
"""

import argparse
import copy
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.dataset_creation_checks as checks
from src.utils import set_seed, load_data, load_rules_amie, display_triple


def generate_negatives(cf, fact, rules, kg, cf_type, symmetric_relations):
    """Generate head, tail, and rel corruptions of a given fact according to our constraints"""
    all_conclusions = []
    corruptions = []

    ##### identify conclusions that cf might trigger according to our rules to avoid false negatives
    filtered_rules = rules[(rules['antecedent_1'] == cf[1]) | (rules['antecedent_2'] == cf[1])]
    assert filtered_rules.shape[0] > 0  # at least original rule should be included (the one that generated our inference)
    if cf_type == 0:  # (x, rel_1, y) is the counterfactual
        rel_1 = cf[1]
        for i, r in filtered_rules.iterrows():
            if rel_1 == r['antecedent_1']:  # x rel_1 y, y r z -> x r' z
                possible_z = np.unique(
                    kg[(kg['relation'] == r['antecedent_2']) & (kg['head'] == cf[2])]['tail']).tolist()
                if len(possible_z) > 0:
                    all_conclusions.extend([[cf[0], r['head'], z] for z in possible_z])

            if rel_1 == r['antecedent_2']:  # z r x, x rel_1 y -> z r' y
                possible_z = np.unique(
                    kg[(kg['relation'] == r['antecedent_1']) & (kg['tail'] == cf[0])]['head']).tolist()
                if len(possible_z) > 0:
                    all_conclusions.extend([[z, r['head'], cf[2]] for z in possible_z])

    elif cf_type == 1:  # (y, rel_2, z) is the counterfactual
        rel_2 = cf[1]
        for i, r in filtered_rules.iterrows():
            if rel_2 == r['antecedent_1']:  # y rel_2 z, z rel x -> y rel_3 x
                possible_x = np.unique(
                    kg[(kg['relation'] == r['antecedent_2']) & (kg['head'] == cf[2])]['tail']).tolist()
                if len(possible_x) > 0:
                    all_conclusions.extend([[cf[0], r['head'], x] for x in possible_x])

            if rel_2 == r['antecedent_2']:  # x rel_1 y, y rel_2 z -> x rel_3 z
                possible_x = np.unique(
                    kg[(kg['relation'] == r['antecedent_1']) & (kg['tail'] == cf[0])]['head']).tolist()
                if len(possible_x) > 0:
                    all_conclusions.extend([[x, r['head'], cf[2]] for x in possible_x])

    #### create corruption of the fact
    # check which entities are eligible
    rel = fact[1]
    rel_data = kg[kg['relation'] == rel]
    all_entities = np.unique(kg['head'].values.tolist() + kg['tail'].values.tolist()).tolist() # all entities in full kg

    # possible head replacements
    possible_heads = np.unique(rel_data['head']).tolist()
    possible_heads.remove(fact[0])  # do not sample same head again
    if len(possible_heads) == 0:
        possible_heads = copy.deepcopy(all_entities)

    # possible tail replacements
    possible_tails = np.unique(rel_data['tail']).tolist()
    possible_tails.remove(fact[2])  # do not sample same tail again
    if len(possible_tails) == 0:
        possible_tails = copy.deepcopy(all_entities)

    # possible relation corruptions (no additional constraints)
    possible_rels = copy.deepcopy(all_rels)
    possible_rels.remove(fact[1])

    head_corr, tail_corr, rel_corr = None, None, None

    while head_corr is None:
        s_head = np.random.choice(possible_heads)
        candidate = [s_head, rel, fact[2]]
        # check that candidate not in KG and not a valid conclusion according to our rules
        if checks._in_kg(kg, candidate, symmetric_relations) is False and \
                checks._in_list(all_conclusions, candidate, symmetric_relations) is False:
            head_corr = candidate
            assert head_corr[0] != fact[0]
        else:
            # do not sample this head again
            possible_heads.remove(s_head)
        if len(possible_heads) == 0:
            # revert to all entities being available
            print('No head corruption found.')
            possible_heads = copy.deepcopy(all_entities)

    while tail_corr is None:
        s_tail = np.random.choice(possible_tails)
        candidate = [fact[0], rel, s_tail]
        # check that candidate not in KG and not a valid conclusion according to our rules
        if checks._in_kg(full_kg, candidate, symmetric_relations) is False and \
                checks._in_list(all_conclusions, candidate, symmetric_relations) is False:
            tail_corr = candidate
            assert tail_corr[2] != fact[2]
        else:
            # do not sample this tail again
            possible_tails.remove(s_tail)
        if len(possible_tails) == 0:
            # revert to all entities being available
            print('No tail corruption found.')
            possible_tails = copy.deepcopy(all_entities)

    while rel_corr is None:
        s_rel = np.random.choice(possible_rels)
        candidate = [fact[0], s_rel, fact[2]]
        # check that candidate not in KG and not a valid conclusion according to our rules
        if checks._in_kg(full_kg, candidate, symmetric_relations) is False and \
                checks._in_list(all_conclusions, candidate, symmetric_relations) is False:
            rel_corr = candidate
            assert rel_corr[1] != fact[1]
        else:
            # do not sample this relation again
            possible_rels.remove(s_rel)
        if len(possible_rels) == 0:
            print('No relation corruption found.')
            return None

    corruptions.append(head_corr)
    corruptions.append(tail_corr)
    corruptions.append(rel_corr)

    return corruptions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="s", type=str)
    parser.add_argument('--rule_set', default='automatic', type=str, choices=['automatic', 'manual'],
                        help="which rule set to use")
    parser.add_argument('--generate', default='near', type=str, choices=['near', 'far'],
                        help="which rule set to use")
    parser.add_argument('--save', default='False', type=str)
    args = parser.parse_args()

    set_seed(0)

    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../.."))
    storing_location_data = os.path.join(path_to_codex_folder, 'cfkgr_data')

    data, symmetric_rels, entity_ids, relation_ids = load_data(path_to_codex_folder, size=args.size)
    full_kg = pd.concat([data.split('train'), data.split('valid'), data.split('test')], ignore_index=True)

    rules = load_rules_amie(path_to_codex_folder, args.size)

    # load dataset that was created so far
    full_data = pd.read_json(os.path.join(storing_location_data, f'cfkgr_{args.size}.json'))
    all_rels = np.unique(data.split('train')['relation']).tolist()

    # make sure naming is consistent across datasets
    if args.size == 'm' and 'og_label' in full_data.columns:
        full_data.rename(columns={'og_label': 'original_label'}, inplace=True)

    print('Number of test instances before negatives are added:', full_data.shape[0])
    full_data_fact = copy.deepcopy(full_data[full_data['type'] == f'{args.generate}_fact'])
    print('Number of facts that require negatives', full_data_fact.shape[0])

    # new data frame with added info
    full_data_new = pd.DataFrame(columns=full_data.columns)

    k = 0 # keep track of dataframe index
    for i, d in tqdm(full_data.iterrows(), total=full_data.shape[0]):
        full_data_new.loc[k, :] = d # add original instance to data frame
        k += 1
        # check if the type of the instance is equal to the type for which we want to generate corruptions
        if d['type'] == f'{args.generate}_fact':
            cf = [d['cf_head'], d['cf_rel'], d['cf_tail']]
            fact = [d['head'], d['rel'], d['tail']]
            corruptions = generate_negatives(cf, fact, rules, full_kg, d['cf_type'], symmetric_rels)
            # insert new rows in dataframe
            new_rows = []
            corr_names = [f'head_corr_{args.generate}', f'tail_corr_{args.generate}', f'rel_corr_{args.generate}']
            for j, c in enumerate(corruptions):
                # copy the original row
                print(display_triple(c, data))
                new_row = copy.deepcopy(d)

                # replace the triple-specific info
                new_row['head'] = c[0]
                new_row['head_libkge'] = entity_ids[c[0]]
                new_row['rel'] = c[1]
                new_row['rel_libkge'] = relation_ids[c[1]]
                new_row['tail'] = c[2]
                new_row['tail_libkge'] = entity_ids[c[2]]
                new_row['type'] = corr_names[j]
                new_row['expected_label'] = 0
                new_row['original_label'] = 0
                full_data_new.loc[k, :] = new_row
                k += 1

    print('Number of instances after:', full_data_new.shape[0])
    if args.save == 'True':
        full_data_new.to_json(os.path.join(storing_location_data, f'cfkgr_{args.size}.json'), orient='records')












