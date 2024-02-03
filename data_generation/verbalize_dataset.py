"""
This file verbalizes a dataset.
"""

import argparse
import ast
import os

import numpy as np
import pandas as pd

from src.utils import load_data, set_seed
from src.verbalize_entry import verbalize_entry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', default='s', type=str, choices=['s', 'm', 'l'], help="size of the CoDEx data")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--save', default='False', type=str)
    args = parser.parse_args()

    # if given, set seed
    if args.seed is not None:
        set_seed(args.seed)

    print('Size:', args.size)
    print('Seed:', args.seed)
    print('\n')

    if args.size == 'm':
        subset = ['head_corr_far', 'rel_corr_far', 'tail_corr_far',
                  'head_corr_near', 'rel_corr_near', 'tail_corr_near']
    else:
        subset = None

    # load data
    path_to_codex_folder = os.path.abspath(os.path.join(__file__, "../.."))
    data, symmetric_rels, entity_ids, relation_ids = load_data(path_to_codex_folder, size=args.size)

    # load test data
    storing_location_data = os.path.join(path_to_codex_folder, 'cfkgr_data')
    test_data = pd.read_csv(os.path.join(storing_location_data, f'cfkgr_{args.size}_test.csv'))
    if 'text' not in test_data.columns:
        test_data['text'] = np.nan # new column

    for i, d in test_data.iterrows():
        if subset is not None:
            if d['type'] in subset:
                verbalization = verbalize_entry(ast.literal_eval(d['test_inst']), data, ast.literal_eval(d['cf']), ast.literal_eval(d['context']), d['type'])
                test_data.loc[i, 'text'] = verbalization
        else:
            # verbalize everything
            verbalization = verbalize_entry(ast.literal_eval(d['test_inst']), data, ast.literal_eval(d['cf']),
                                            ast.literal_eval(d['context']), d['type'])
            test_data.loc[i, 'text'] = verbalization

    if args.save == 'True':
        test_data.to_csv(os.path.join(storing_location_data, f'cfkgr_{args.size}_test.csv'), index=False)



