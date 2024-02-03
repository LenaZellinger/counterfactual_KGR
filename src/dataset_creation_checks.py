"""This file contains some checks used in the data creation."""

import numpy as np
import pandas as pd


def _in_kg(full_kg: pd.DataFrame, triple: list, symmetric_relations: list):
    """checks whether a given triple is present in kg"""
    check = int(((full_kg['head'] == triple[0]) &
                 (full_kg['relation'] == triple[1]) &
                 (full_kg['tail'] == triple[2])).any())

    # make sure equivalent symmetric entry also not in Kg
    if triple[1] in symmetric_relations:
        check += int(((full_kg['head'] == triple[2]) &
                      (full_kg['relation'] == triple[1]) &
                      (full_kg['tail'] == triple[0])).any())
    if check == 0:
        return False
    else:
        return True


def _in_list(triple_list: list, triple: list, symmetric_relations: list):
    """checks whether a given triple is present in a list"""
    if triple[1] not in symmetric_relations:
        return triple in triple_list
    else:
        # check whether symmetric triple is also not in list
        in_list = triple in triple_list
        in_list_sym = triple[::-1] in triple_list
        if in_list is False and in_list_sym is False: # neither original triple nor symmetric counterpart are in list
            return False
        else:
            return True

def _get_heads(kg, r):
    """retrieve all head entities of a relation"""
    return np.unique(kg[kg['relation'] == r]['head'].values).tolist()


def _get_tails(kg, r):
    """retrieve all tail entities of a relation"""
    return np.unique(kg[kg['relation'] == r]['tail'].values).tolist()


def _type_check_entity_pair(data, e_1, e_2):
    """check if there is at least one overlapping type for e_1 and e_2"""
    if len(set(data.entity_types(e_1)) & set(data.entity_types(e_2))) != 0:
        return True
    else:
        return False