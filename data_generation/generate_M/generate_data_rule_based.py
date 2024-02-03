"""
This file contains the code for generating the dataset based on CoDEx-M.
It's structure slightly differs from S and L since it was created earlier due to the human annotation performed on this subset.
However, the creation of the instances is equivalent to the one used for S and L.
We provide this file since it results in the exact examples provided in our dataset.
"""

import argparse
import copy
import json
import os

import numpy as np
from tqdm import tqdm

from dataset_creation import CFGeneration
from src.utils import load_data, set_seed, load_rules_amie, display_triple
from verbalizations import create_verbalized_entry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='m', type=str, choices=['s', 'm', 'l'], help="size of the CoDEx data")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--max_examples', default=25, type=int, help="max. number of examples per rule and atom")
    parser.add_argument('--save', default='False', type=str)
    args = parser.parse_args()

    # if given, set seed
    if args.seed is not None:
        set_seed(args.seed)

    print('Size:', args.size)
    print('Max. examples:', args.max_examples)
    print('Seed:', args.seed)

    # load data
    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../../.."))
    data, symmetric_rels, entity_ids, relation_ids = load_data(path_to_codex_folder, size=args.size)

    # define storing locations
    if args.save == 'True':
        # for annotation
        storing_location = os.path.join(path_to_codex_folder, 'cfkgr_data')
        os.makedirs(storing_location, exist_ok=True)

    # load rules: automatically extracted valid rules
    rules = load_rules_amie(path_to_codex_folder, args.size)

    n_rules = rules.shape[0]
    dataset_annotation = []
    rules_matched = []

    # generate examples
    already_generated = [] # counterfactuals that were already generated across rules
    dataset_generator = CFGeneration(data, rules, symmetric_rels)
    example_id = 0
    for rule_id, r in tqdm(rules.iterrows(), total=rules.shape[0]):
        already_generated_rule = [] # counterfactuals that were already generated for this rule
        print('Current rule:', f'{data.relation_label(r[0])}, {data.relation_label(r[1])} -> {data.relation_label(r[2])}')
        for type in [0, 1]:
            print('CF type:', type)
            was_None = False # initialize was_None as False
            for n in range(args.max_examples):
                if was_None is False:
                    cf, conclusion, context, facts_to_retain_near, facts_to_retain_far, corruptions, og_entities = dataset_generator.get_entry(r, already_generated, type)
                    # no possible cf according to our constraints that was not done yet
                    if cf is None:
                        was_None = True # set was_None to True
                if was_None is True: # was_None True -> try to get unique cf for rule
                    print('No more unique counterfactuals. Allowing for repeated cf now.')
                    cf, conclusion, context, facts_to_retain_near, facts_to_retain_far, corruptions, og_entities = dataset_generator.get_entry(r, already_generated_rule, type)
                if cf is None: # if both failed -> break
                    break
                else:
                    already_generated_rule.append(copy.deepcopy(cf))

                if conclusion is not None:
                    already_generated.append(copy.deepcopy(cf))
                    assert len(facts_to_retain_near) <= 2

                    rules_matched.append(rule_id)

                    print('Counterfactual:', display_triple(cf, data))
                    print('Context:', display_triple(context, data))
                    print('Assertion:', display_triple(conclusion, data))
                    for f in facts_to_retain_near:
                        print('Near fact to retain:', display_triple(f, data))
                    for f in facts_to_retain_far:
                        print('Far fact to retain:', display_triple(f, data))
                    print('Head corruption:', display_triple(corruptions[0], data))
                    print('Tail corruption:', display_triple(corruptions[1], data))
                    print('Relation corruption:', display_triple(corruptions[2], data))
                    print('\n')

                    dataset_annotation = create_verbalized_entry(data,
                                                                 dataset_annotation,
                                                                 cf,
                                                                 context,
                                                                 conclusion,
                                                                 corruptions,
                                                                 og_entities,
                                                                 facts_to_retain_near,
                                                                 facts_to_retain_far,
                                                                 rule_id,
                                                                 example_id,
                                                                 args.size,
                                                                 type,
                                                                 style='prompt')

                    example_id += 1

    if args.save == 'True':
        json.dump(dataset_annotation, open(os.path.join(storing_location, f'cfkgr_{args.size}.json'), 'w'))

    # Rule distribution
    print('Rule match counts:', np.bincount(rules_matched, minlength=n_rules))
    rule_dist = np.bincount(rules_matched, minlength=n_rules) / example_id
    print('Rule match distribution:', np.round(rule_dist, 3))
    assert np.isclose(np.sum(rule_dist), 1)
    print(f'Total data entries created: {example_id}')

    # Initialize a set to store unique sublists
    unique_sublists = set()

    for sublist in already_generated:
        sublist_tuple = tuple(sublist)
        unique_sublists.add(sublist_tuple)

    # Get the number of unique sublists
    num_unique_sublists = len(unique_sublists)

    print(f'Number of unique counterfactuals: {num_unique_sublists}')