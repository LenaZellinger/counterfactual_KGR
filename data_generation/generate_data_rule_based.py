"""This file creates the CFKGR datasets used in our experiments based on the CoDEx knowledge graphs."""

import argparse
import copy
import json
import os

import numpy as np
from tqdm import tqdm

from src.dataset_creation import CFGeneration
from src.utils import load_data, set_seed, load_rules_amie, display_triple, create_libkge_entry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='s', type=str, choices=['s', 'm', 'l'], help="size of the CoDEx data")
    parser.add_argument('--seed', default=0, type=int, help="seed 0 is used for our experiments")
    parser.add_argument('--max_examples', default=25, type=int, help="max. number of examples per rule and atom; 25 in our experiments")
    parser.add_argument('--save', default='True', type=str, choices=['True', 'False'])
    args = parser.parse_args()

    # if given, set seed
    if args.seed is not None:
        set_seed(args.seed)

    # print current setting
    print('Size:', args.size)
    print('Max. examples:', args.max_examples)
    print('Seed:', args.seed)

    # load data
    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../.."))
    data, symmetric_rels, entity_ids, relation_ids = load_data(path_to_codex_folder, size=args.size)

    # define storing locations
    if args.save == 'True':
        storing_location_data = os.path.join(path_to_codex_folder, 'cfkgr_data')
        os.makedirs(storing_location_data, exist_ok=True)

    # load rules
    rules = load_rules_amie(path_to_codex_folder, args.size)
    n_rules = rules.shape[0]

    # create objects to keep track of data generation
    dataset_train = []
    rules_matched = []
    already_generated = []
    example_id = 0

    # initialize the dataset generator
    dataset_generator = CFGeneration(data, rules, symmetric_rels)

    # generate the dataset by looping over rules
    for rule_id, r in tqdm(rules.iterrows(), total=rules.shape[0]):
        already_generated_rule = [] # keep track of counterfactuals that were already generated for this rule
        print('Current rule:', f'{data.relation_label(r[0])}, {data.relation_label(r[1])} -> {data.relation_label(r[2])}')

        for type in [0, 1]: # 0 -> first atom changed, 1 -> second atom changed
            print('CF type:', type)
            total_created = 0 # keep track of examples for this atom
            was_None = False # keep track whether the construction of a NEW cf across all rules was impossible under our constraints

            for n in range(args.max_examples):
                if was_None is False: # when None was not returned yet
                    cf, conclusion, context, facts_to_retain_near, facts_to_retain_far, corruptions, corruptions_far = dataset_generator.get_entry(r, already_generated, type)
                    if cf is None: # no possible new cf according to our constraints across the entire dataset
                        was_None = True

                if was_None is True: # try to get a unique cf for this rule at least
                    print('No more unique counterfactuals. Allowing for repeated cf in the dataset now.')
                    cf, conclusion, context, facts_to_retain_near, facts_to_retain_far, corruptions, corruptions_far = dataset_generator.get_entry(r, already_generated_rule, type)

                if cf is None: # if both failed -> move on to next cf type
                    break
                else:
                    already_generated_rule.append(copy.deepcopy(cf))

                # if full suitable test case is found
                if conclusion is not None:
                    # print instance
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

                    # append instance to dataset
                    dataset_train = create_libkge_entry(dataset_train,
                                                        cf,
                                                        context,
                                                        conclusion,
                                                        corruptions,
                                                        corruptions_far,
                                                        facts_to_retain_near,
                                                        facts_to_retain_far,
                                                        rule_id,
                                                        example_id,
                                                        args.size,
                                                        type,
                                                        entity_ids,
                                                        relation_ids)

                    example_id += 1
                    total_created += 1
                    rules_matched.append(rule_id)
                    already_generated.append(copy.deepcopy(cf)) # mark instance as already generated

    # save the dataset
    if args.save == 'True':
        json.dump(dataset_train, open(os.path.join(storing_location_data, f'cfkgr_{args.size}.json'), 'w'))

    # print statistics
    print('Rule match counts:', np.bincount(rules_matched, minlength=n_rules))
    rule_dist = np.bincount(rules_matched, minlength=n_rules) / example_id
    print('Rule match distribution:', np.round(rule_dist, 3))
    assert np.isclose(np.sum(rule_dist), 1)
    print(f'Total data entries created: {example_id}')