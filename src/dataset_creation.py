import copy

import numpy as np
import pandas as pd

import src.dataset_creation_checks as checks


class CFGeneration:
    def __init__(self, data, rules, symmetric_relations):
        self.data = data
        self.train = data.split('train')
        self.valid = data.split('valid')
        self.test = data.split('test')
        self.full_kg = pd.concat([self.train, self.valid, self.test], ignore_index=True)
        assert self.full_kg.shape[0] == self.full_kg.drop_duplicates().shape[0]
        self.rules = rules
        self.symmetric_relations = symmetric_relations
        self.all_rels = np.unique(self.train['relation']).tolist()

    def get_entry(self, rule, already_generated, type):
        """create an entry for the CCKG dataset"""
        self.cf = None
        self.conclusion = None
        self.context = None
        self.corruptions = []
        self.corruptions_far = []
        self.og_triples = []
        self.all_conclusions = []
        self.already_generated = already_generated

        ### sample a candidate counterfactual, it's conclusions, and the used context from the training KG
        self._sample_cf(rule, type, self.train) # establishes self.cf, self.context, self.conclusion
        if self.cf is None:
            return None, None, None, None, None, None, None

        ### collect all other potential conclusions of the cf given our rule set to avoid false negatives
        self._get_conclusions(self.full_kg, type) # establishes self.all_conclusions

        ### collect facts that are unaffected by cf (according to our assumptions)
        selected_fact_near, selected_fact_far = self._get_facts_to_retain(self.full_kg, n_near=2, n_far=1)

        # create additional negatives via corruption for conclusion and far facts
        self._get_corruptions(self.full_kg, self.conclusion, self.corruptions)
        self._get_corruptions(self.full_kg, selected_fact_far[0], self.corruptions_far)

        # cf, conclusion, context, facts_to_retain_near, facts_to_retain_far, corruptions, corruptions_far
        return self.cf, self.conclusion, self.context, selected_fact_near, selected_fact_far, self.corruptions, self.corruptions_far


    def _sample_cf(self, rule, type: int, kg: pd.DataFrame):
        """create the counterfactual triple"""
        rel_1 = rule['antecedent_1']
        rel_2 = rule['antecedent_2']
        rel_3 = rule['head']

        # if no type given randomly choose whether to corrupt first or second link
        if type is None:
            raise ValueError('It is not specified, which atom of the rule should be the counterfactual. Please set type=0 (for the first) or type=1 (for the second)')

        if type == 0: # x rel_1 y is the counterfactual
            ### search for a potential connection y rel_2 z that fulfils our constraints:
            # y is a tail of r_1
            # z is a tail of r_3
            possible_links_2 = kg[(kg['relation'] == rel_2) &
                                    (kg['head'].isin(checks._get_tails(self.full_kg, rel_1))) &
                                    (kg['tail'].isin(checks._get_tails(self.full_kg, rel_3)))]
            possible_links_2.reset_index(drop=True, inplace=True)

            if possible_links_2.shape[0] == 0:
                print("No suitable existing link can be found.")
                return

            # shuffle the list of candidates
            possible_links_2 = possible_links_2.sample(frac=1)

            ### identify the set of potential links for the counterfactual x rel_1 y that fulfil our constraints:
            # x is a head of r_3
            # no constraint on y since it will be replaced
            possible_links_1 = kg[(kg['relation'] == rel_1) &
                                    (kg['head'].isin(checks._get_heads(self.full_kg, rel_3)))]
            possible_links_1.reset_index(drop=True, inplace=True)

            if possible_links_1.shape[0] == 0:
                print("No suitable existing link can be found for the cf.")
                return

            ### Check if a suitable counterfactual x rel_1 y can be created from the available links
            for _, k in possible_links_2.iterrows(): # for all existing links (y, r_2, z)
                y = k[0]
                z = k[2]
                assert rel_2 == k[1]

                possible_links_1 = possible_links_1.sample(frac=1) # shuffle potential counterfactual links
                for i, l in possible_links_1.iterrows(): # counterfactual link (x, r_1, y')
                    x = l[0]
                    cf = [x, rel_1, y]
                    assert rel_1 == l[1]

                    ### check whether cf and conclusion fulfil all constraints
                    # additional check for type of/member of
                    if rel_1 in ['P361', 'P463']:
                        if checks._type_check_entity_pair(self.data, l[2], y) is False: # check that replacement has at least one overlapping type
                            continue
                    # check that cf not in KG
                    if checks._in_kg(self.full_kg, cf, self.symmetric_relations) is True: # if already in KG -> move on
                        continue
                    # check that cf was not created already
                    if checks._in_list(self.already_generated, cf, self.symmetric_relations) is True: # if cf was already generated -> move on
                        continue
                    conclusion = [x, rel_3, z]
                    # check that conclusion is not in KG
                    if checks._in_kg(self.full_kg, conclusion, self.symmetric_relations) is True: # if conclusion in KG -> move on
                        continue

                    ### if all checks are passed, record data entry
                    self.conclusion = conclusion
                    self.context = [y, rel_2, z]
                    self.cf = cf

                    # assert that context is stored correctly
                    assert self.context[0] == k[0]
                    assert self.context[1] == k[1]
                    assert self.context[2] == k[2]
                    assert checks._in_kg(self.full_kg, self.context, self.symmetric_relations) is True

                    # break loop if suitable conclusion is found
                    if self.conclusion is not None:
                        break

                # break loop if suitable conclusion is found
                if self.conclusion is not None:
                    break

            if self.conclusion is None:
                print('Could not connect links.')

        elif type == 1:
            ### search for a potential connection x rel_1 y that fulfils our constraints:
            # y is a head of r_2
            # x is a head of r_3
            possible_links_1 = kg[(kg['relation'] == rel_1) &
                                    (kg['tail'].isin(checks._get_heads(self.full_kg, rel_2))) &
                                    (kg['head'].isin(checks._get_heads(self.full_kg, rel_3)))]
            possible_links_1.reset_index(drop=True, inplace=True)

            if possible_links_1.shape[0] == 0:
                print("No suitable existing links found.")
                return None

            # shuffle the remaining links
            possible_links_1 = possible_links_1.sample(frac=1)

            ### identify the set of potential links for the counterfactual y rel_2 z that fulfil our constraints:
            # z is a tail of r_3
            # no constraint on y since it will be replaced
            possible_links_2 = kg[(kg['relation'] == rel_2) &
                                    (kg['tail'].isin(checks._get_tails(self.full_kg, rel_3)))]
            possible_links_2.reset_index(drop=True, inplace=True)

            if possible_links_2.shape[0] == 0:
                print("No suitable existing links found for cf.")
                return None

            ### Check if a suitable counterfactual y rel_2 z can be created for any of the available links
            for _, k in possible_links_1.iterrows(): # for all existing links (x, rel_1, y')
                x = k[0]
                y = k[2]
                assert rel_1 == k[1]

                possible_links_2 = possible_links_2.sample(frac=1) # shuffle potential counterfactual links
                for i, l in possible_links_2.iterrows(): # counterfactual link (y, rel_2, z)
                    z = l[2]
                    cf = [y, rel_2, z]
                    assert rel_2 == l[1]

                    ### check whether cf and conclusion fulfil all constraints
                    # additional check for type of/member of
                    if rel_2 in ['P361', 'P463']:
                        if checks._type_check_entity_pair(self.data, l[0], y) is False:
                            continue
                    # check that cf not in KG
                    if checks._in_kg(self.full_kg, cf, self.symmetric_relations) is True:
                        continue
                    # check that cf was not created already
                    if checks._in_list(self.already_generated, cf, self.symmetric_relations) is True:  # if already in KG -> move on
                        continue
                    conclusion = [x, rel_3, z]
                    # check that conclusion is not in KG
                    if checks._in_kg(self.full_kg, conclusion, self.symmetric_relations) is True: # if conclusion in KG -> move on
                        continue

                    # if all checks are passed, record data entry
                    self.conclusion = conclusion
                    self.context = [x, rel_1, y]
                    self.cf = cf

                    # assert that context is stored correctly
                    assert self.context[0] == k[0]
                    assert self.context[1] == k[1]
                    assert self.context[2] == k[2]
                    assert checks._in_kg(self.full_kg, self.context, self.symmetric_relations) is True

                    # break loop if suitable conclusion is found
                    if self.conclusion is not None:
                        break

                # break loop if suitable conclusion is found
                if self.conclusion is not None:
                    break

            if self.conclusion is None:
                print('Could not connect links.')

        else:
            raise ValueError('Only atom 0 and atom 1 can attain a counterfactual instantiation.')


    def _get_conclusions(self, kg, type):
        ### retrieve all conclusions from the rule set given the counterfactual
        filtered_rules = self.rules[(self.rules['antecedent_1'] == self.cf[1]) | (self.rules['antecedent_2'] == self.cf[1])]
        assert filtered_rules.shape[0] > 0 # at least original rule should be included

        if type == 0: # (x, rel_1, y) is the counterfactual
            rel_1 = self.cf[1]
            for i, r in filtered_rules.iterrows():
                if rel_1 == r['antecedent_1']:  # x rel_1 y, y r z -> x r' z
                    possible_z = np.unique(kg[(kg['relation'] == r['antecedent_2']) & (kg['head'] == self.cf[2])]['tail']).tolist()
                    if len(possible_z) > 0:
                        self.all_conclusions.extend([[self.cf[0], r['head'], z] for z in possible_z])

                if rel_1 == r['antecedent_2']:  # z r x, x rel_1 y -> z r' y
                    possible_z = np.unique(kg[(kg['relation'] == r['antecedent_1']) & (kg['tail'] == self.cf[0])]['head']).tolist()
                    if len(possible_z) > 0:
                        self.all_conclusions.extend([[z, r['head'], self.cf[2]] for z in possible_z])

            assert self.conclusion in self.all_conclusions

        elif type == 1: # (y, rel_2, z) is the counterfactual
            rel_2 = self.cf[1]
            for i, r in filtered_rules.iterrows():
                if rel_2 == r['antecedent_1']: # y rel_2 z, z rel x -> y rel_3 x
                    possible_x = np.unique(kg[(kg['relation'] == r['antecedent_2']) & (kg['head'] == self.cf[2])]['tail']).tolist()
                    if len(possible_x) > 0:
                        self.all_conclusions.extend([[self.cf[0], r['head'], x] for x in possible_x])

                if rel_2 == r['antecedent_2']: # x rel_1 y, y rel_2 z -> x rel_3 z
                    possible_x = np.unique(kg[(kg['relation'] == r['antecedent_1']) & (kg['tail'] == self.cf[0])]['head']).tolist()
                    if len(possible_x) > 0:
                        self.all_conclusions.extend([[x, r['head'], self.cf[2]] for x in possible_x])

            # assert that our selected conclusion was constructed
            assert self.conclusion in self.all_conclusions

        else:
            raise ValueError('Invalid type! Choose 0 for first atom, and 1 for second atom.')


    def _get_facts_to_retain(self, kg, n_near, n_far):
        """facts that are unaffected by our counterfactual under current assumptions"""

        ### select two near facts
        near_triples = kg[(kg['head'].isin([self.cf[0], self.cf[2]])) | (kg['tail'].isin([self.cf[0], self.cf[2]]))] # 'close' facts
        near_triples_wo_context = near_triples[~((near_triples['head'] == self.context[0]) &
                                    (near_triples['relation'] == self.context[1]) &
                                    (near_triples['tail'] == self.context[2]))] # do not sample context
        assert near_triples_wo_context.shape[0] == near_triples.shape[0] - 1

        if self.context[1] in self.symmetric_relations:
            near_triples_wo_context = near_triples_wo_context[~((near_triples_wo_context['head'] == self.context[2]) &
                                          (near_triples_wo_context['relation'] == self.context[1]) &
                                          (near_triples_wo_context['tail'] == self.context[0]))]

        # randomly sample max. n_near closest facts
        selected_fact_near = near_triples_wo_context.sample(n=min(n_near, near_triples_wo_context.shape[0]), replace=False)
        selected_fact_near = selected_fact_near.values.tolist()

        ### select one far fact
        # random fact from kg that is not in the 1-hop neighborhood of cf
        far_triples = kg.merge(near_triples, how='left', indicator=True)
        far_triples = far_triples[far_triples['_merge'] == 'left_only']
        far_triples = far_triples[['head', 'relation', 'tail']]

        # make sure no near triples are included anymore
        assert self.cf[0] not in np.unique(far_triples['head']).tolist()
        assert self.cf[2] not in np.unique(far_triples['head']).tolist()
        assert self.cf[0] not in np.unique(far_triples['tail']).tolist()
        assert self.cf[2] not in np.unique(far_triples['tail']).tolist()
        assert near_triples.shape[0] + far_triples.shape[0] == kg.shape[0]

        selected_fact_far = far_triples.sample(n=min(n_far, far_triples.shape[0]), replace=False)
        selected_fact_far = selected_fact_far.values.tolist()

        return selected_fact_near, selected_fact_far

    def _get_corruptions(self, kg: pd.DataFrame, consequence: list, add_list: list):
        """Produce head, tail and relation corruptions"""
        rel = consequence[1]
        rel_data = kg[kg['relation'] == rel]
        all_entities = np.unique((self.full_kg['head'].values.tolist() + self.full_kg['tail'].values.tolist())).tolist()

        #### define potential head replacements
        possible_heads = np.unique(rel_data['head']).tolist() # head of the same relation
        possible_heads.remove(consequence[0]) # do not sample same head again # redundant
        if len(possible_heads) == 0:
            possible_heads = copy.deepcopy(all_entities) # default to all entities if resulting set empty

        #### define potential tail replacements
        possible_tails = np.unique(rel_data['tail']).tolist() # tail of the same relation
        possible_tails.remove(consequence[2]) # do not sample same tail again # redundant
        if len(possible_tails) == 0:
            possible_tails = copy.deepcopy(all_entities)

        # possible relation corruptions (no additional constraints)
        possible_rels = copy.deepcopy(self.all_rels)
        possible_rels.remove(consequence[1])  # do not sample same rel again

        head_corr, tail_corr, rel_corr = None, None, None

        ### choose a head corruption
        while head_corr is None:
            s_head = np.random.choice(possible_heads)
            candidate = [s_head, rel, consequence[2]]
            # check that candidate not in KG and not a valid conclusion according to our rules
            if checks._in_kg(self.full_kg, candidate, self.symmetric_relations) is False and \
                    checks._in_list(self.all_conclusions, candidate, self.symmetric_relations) is False:
                head_corr = candidate
            else:
                # do not sample this head again
                possible_heads.remove(s_head)
            if len(possible_heads) == 0:
                # default to the full entity set
                print('No head corruption found. Using all entities now.')
                possible_heads = copy.deepcopy(all_entities)
        add_list.append(head_corr)

        while tail_corr is None:
            s_tail = np.random.choice(possible_tails)
            candidate = [consequence[0], rel, s_tail]
            # check that candidate not in KG and not a valid conclusion according to our rules
            if checks._in_kg(self.full_kg, candidate, self.symmetric_relations) is False and \
                    checks._in_list(self.all_conclusions, candidate, self.symmetric_relations) is False:
                tail_corr = candidate
            else:
                # do not sample this tail again
                possible_tails.remove(s_tail)
            if len(possible_tails) == 0:
                # default to the full entity set
                print('No head corruption found. Using all entities now.')
                possible_tails = copy.deepcopy(all_entities)
        add_list.append(tail_corr)

        while rel_corr is None:
            s_rel = np.random.choice(possible_rels)
            candidate = [consequence[0], s_rel, consequence[2]]
            # check that candidate not in KG and not a valid conclusion according to our rules
            if checks._in_kg(self.full_kg, candidate, self.symmetric_relations) is False and \
                    checks._in_list(self.all_conclusions, candidate, self.symmetric_relations) is False:
                rel_corr = candidate
            else:
                # do not sample this relation again
                possible_rels.remove(s_rel)
            # this should never be triggered
            if len(possible_rels) == 0:
                print('No relation corruption found.')
                self.conclusion = None
                return
        add_list.append(rel_corr)
