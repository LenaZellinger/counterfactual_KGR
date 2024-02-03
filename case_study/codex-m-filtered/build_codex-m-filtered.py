########################################################################################################################
# This file builds the filtered test set for CoDEx-M based on Amie3 rules
########################################################################################################################

import argparse
import os

import numpy as np
from tqdm import tqdm

from src.utils import load_data, set_seed, load_rules_amie

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='m', type=str, choices=['s', 'm', 'l'], help="size of the CoDEx data")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save', default='True', type=str)
    args = parser.parse_args()

    # if given, set seed
    if args.seed is not None:
        set_seed(args.seed)

    # load data
    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../../.."))
    results_folder = os.path.join(path_to_codex_folder, 'case_study', 'codex-m-filtered')
    os.makedirs(results_folder, exist_ok=True)
    data, symmetric_rels, entity_ids, relation_ids = load_data(path_to_codex_folder, size=args.size)

    # load rules
    rules = load_rules_amie(path_to_codex_folder, args.size)

    # load train and test data
    train_data = data.split('train')
    test_data = data.split('test')

    test_rels = np.unique(rules['head'].values).tolist() # relations that can be inferred via rules
    test_data['is_inferential'] = False
    test_data['rules_matched'] = test_data.apply(lambda x: [], axis=1)
    rule_dist = dict(zip(range(rules.shape[0]), np.zeros(rules.shape[0]))) # keep track of how often rule is triggered

    print('Looking for rule matches:')
    for r in test_rels:
        # test data with given relation and the rules that can infer it
        r_data = test_data[test_data['relation'] == r]
        rules_for_r = rules[rules['head'] == r]

        for instance_index, test_entry in tqdm(r_data.iterrows(), total=r_data.shape[0]): # go through all triples that could be inferred
            for rule_index, current_rule in rules_for_r.iterrows(): # go through every rule that could infer them
                # rules ALL have the shape: (x, rel_1, y) and (y, rel_2, z) -> (x, rel_3, z)
                links_1 = train_data[(train_data['head'] == test_entry['head']) & (train_data['relation'] == current_rule['antecedent_1'])]
                links_2 = train_data[(train_data['tail'] == test_entry['tail']) & (train_data['relation'] == current_rule['antecedent_2'])]
                # if we found to links, check whether they are connected by sharing y
                if len(list(set(links_1['tail'].values) & set(links_2['head'].values))) != 0:
                    rule_dist[rule_index] += 1 # update the number of times a rule inferred a triple
                    test_data.iloc[instance_index, -2] = True # set the 'inferential' status of the corresponding triple to True
                    test_data.iloc[instance_index, -1].append(rule_index) # append the rule index to the rules that infer the triple

    # filter dataset to only contain inferential triples inferred by rules with at least 5 matches
    filtered_test_set = test_data[test_data['is_inferential']]
    print('Percentage inferential:', filtered_test_set.shape[0] / test_data.shape[0])
    print('Number of inferential examples (total):', filtered_test_set.shape[0], "\n")

    selected_rules = [k for k in rule_dist.keys() if rule_dist[k] >= 5]
    final_df = filtered_test_set[filtered_test_set['rules_matched'].apply(lambda x: any(item in selected_rules for item in x))]
    print('Numer of rules (after rule filtering):', len(selected_rules))
    print('Number of inferential examples (after rule filtering):', final_df.shape[0])

    # save filtered test set
    final_df.reset_index(drop=True, inplace=True)
    if args.save == 'True':
        final_df.to_csv(os.path.join(results_folder, f'codex-m-filtered.csv'))
