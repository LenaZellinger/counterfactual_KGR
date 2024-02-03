import numpy as np

rel_map_context = {
    'P108': 'is employed by', # employer # checked
    'P112': 'is founded by', # founded by # checked
    'P119': 'is buried in', # place of burial # checked
    'P131': 'is located in the administrative territorial entity', # located in the administrative territorial entity # checked
    'P138': 'is named after', # named after # checked
    'P140': 'belongs to the religion or is associated with the religion', # religion or worldview (but in CoDEx it's titled religion) # checked
    'P1412': 'speaks, writes or signs', # languages spoken, written or signed # checked
    'P1454': 'has the legal form of a', # legal form # checkmark
    'P159': 'is headquartered in', # headquarters location # checked
    'P17': 'is in the country', # country # checked
    'P170': 'was created by', # creator
    'P172': 'belongs to the ethnic group', # ethnic group
    'P19': 'was born in', # place of birth
    'P20': 'died in', # place of death
    'P2348': 'occurred in the time period', # time period
    'P26': 'is married to', # spouse
    'P264': 'belongs to the record label', # record label
    'P27': 'is a citizen of', # country of citizenship
    'P30': 'is part of the continent', # continent
    'P3095': 'is practiced by', # practiced by
    'P3373': 'is the sibling of', # sibling
    'P361': 'is part of', # part of
    'P451': 'is the unmarried partner of', # unmarried partner
    'P452': 'is in the industry', # industry
    'P4552': 'belongs to the mountain range', # mountain range
    'P463': 'is a member of', # member of
    'P47': 'shares a border with', # shares border with
    'P495': 'originated in the country', # country of origin
    'P530': 'has a diplomatic relation with', # diplomatic relation
    'P54': 'is a member of the sports team', # member of sports team
    'P551': 'resides in', # residence
    'P641': 'participates in the sport or is associated with the sport', # sport
    'P69': 'is or was educated at',
    'P740': 'was formed in', # location of formation
    'P840': 'has the narrative location', # narrative location
    'P101': 'works in', # field of work
    'P102': 'is a member of the political party or is affiliated with the political party', # member of political party
    'P1050': 'has the medical condition', # medical condition
    'P1056': 'produces the product or material', # product or material produced or service provided
    'P114': 'belongs to the airline alliance', # airline alliance
    'P1303': 'plays the instrument', # instrument
    'P135': 'is associated with the movement', # movement
    'P136': 'is associated with the genre', # genre
    'P149': 'has the architectural style', # architectural style
    'P194': 'is governed by', # legislative body
    'P1995': 'belongs to the medical field of', # health specialty
    'P2283': 'uses', # uses
    'P2578': 'studies', # is the study of (but in codex studies)
    'P457': 'has the foundational text', # foundational text
    'P509': 'died due to', # cause of death
    'P57': 'was directed by', # director
    'P737': 'is influenced by', # influenced by
    'P366': 'is used for', # has use
    'P106': 'has the occupation' # occupation
}


reverse_rel_map_context = {
    'P161': 'was cast in', # cast member
    'P36': 'is the capital of', # capital
    'P37': 'is the official language of', # official language
    'P40': 'is the child of', # child
    'P749': 'is the parent organization of', # parent organization
    'P113': 'serves as a hub for', # airline hub
    'P169': 'is the chief executive officer of', # chief executive officer
    'P2176': 'is a drug or therapy used to treat', # drug or therapy used for treatment
    'P35': 'is the head of state of', # head of state
    'P488': 'is the chairperson of', # chairperson
    'P84': 'is the architect of', # architect
    'P123': 'is the publisher of', # publisher
    'P466': 'occupies', # occupant
    'P780': 'is a symptom or sign of', # symptoms and signs
    'P800': 'is a notable work of', # notable work
    'P50': 'is the author of',  # author
    'P407': 'is the language of the work or name' # PARAREL language of work or name; language associated with this creative work (such as books, shows, songs, broadcasts or websites) or a name
}

# alternatively: add 3 random types per entity? or too cumbersome to read?

def verbalize_triple(data, triple, rel_mapping, reverse_rel_mapping, type_info=True):

    ##### add additional info
    additional_info = ['', '']
    if type_info is True:
        for i, e in enumerate([triple[0], triple[2]]):
            entity_types = [data.entity_type_label(t) for t in data.entity_types(e)]

            if len(entity_types) > 3: # at most 3 entity types
                entity_types = np.random.choice(entity_types, 3, replace=False)

            for t in entity_types:
                additional_info[i] = append_info(additional_info[i], t)

            # close bracket
            if len(additional_info[i]) > 0:
                additional_info[i] = additional_info[i] + ')'

    if triple[1] in rel_mapping.keys():
        string = f'{data.entity_label(triple[0])} {additional_info[0]} {rel_mapping[triple[1]]} {data.entity_label(triple[2])} {additional_info[1]}'.strip()
        string = string.replace("  ", " ")
        return string

    elif triple[1] in reverse_rel_mapping.keys():
        string = f'{data.entity_label(triple[2])} {additional_info[1]} {reverse_rel_mapping[triple[1]]} {data.entity_label(triple[0])} {additional_info[0]}'.strip()
        string = string.replace("  ", " ")
        return string

    else:
        raise ValueError('No corresponding entry.')

def append_info(additional_info: str, string: str):
    """aappend types to additional info string"""
    if len(additional_info) == 0:
        additional_info += f'({string}'
    else:
        additional_info += f', {string}'
    return additional_info


def create_verbalized_entry(data,
                            dataset,
                            cf,
                            context,
                            conclusion,
                            corruptions,
                            og_entities,
                            facts_to_retain_near,
                            facts_to_retain_far,
                            rule_id,
                            example_id,
                            size,
                            type,
                            style='prompt'):
    """Create dataset entry. Near and far fact corruptions are not included yet and will be added in the file 'create_additional_negatives.py'"""
    if style == 'prompt':
        # verbalize triples for annotation/gpt experiments
        cf_verb = verbalize_triple(data, cf, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
        context_verb = verbalize_triple(data, context, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
        conclusion_verb = verbalize_triple(data, conclusion, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
        head_corr_verb = verbalize_triple(data, corruptions[0], rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
        head_og = verbalize_triple(data, og_entities[0], rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
        tail_corr_verb = verbalize_triple(data, corruptions[1], rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
        tail_og = verbalize_triple(data, og_entities[1], rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
        rel_corr_verb = verbalize_triple(data, corruptions[2], rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
        near_facts = []
        for f in facts_to_retain_near:
            near_facts.append(verbalize_triple(data, f, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context))
        far_facts = []
        for f in facts_to_retain_far:
            far_facts.append(verbalize_triple(data, f, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context))

        prompt = f'Hypothetical scenario: {cf_verb}\n\nContext: {context_verb}\n\n'

        conclusion_str = prompt + f'Is it then plausible that {conclusion_verb}, given that this is not the case in the real world?'
        head_corruption_str = prompt + f'Is it then plausible that {head_corr_verb}, given that this is not the case in the real world?'
        tail_corruption_str = prompt + f'Is it then plausible that {tail_corr_verb}, given that this is not the case in the real world?'
        rel_corruption_str = prompt + f'Is it then plausible that {rel_corr_verb}, given that this is not the case in the real world?'

        near_fact_verb_str = []
        for f in near_facts:
            near_fact_verb_str.append(prompt + f'Is it then still plausible that {f}, given that this is the case in the real world?')

        far_fact_verb_str = []
        for f in far_facts:
            far_fact_verb_str.append(prompt + f'Is it then still plausible that {f}, given that this is the case in the real world?')

        dataset.append({'text': conclusion_str,
                        'head': conclusion[0], 'rel': conclusion[1], 'tail': conclusion[2],
                        'rule': rule_id, 'id': example_id, 'type': 'conclusion',
                        'expected_label': 1,  'og_label': 0,
                        'cf_head': cf[0], 'cf_rel': cf[1], 'cf_tail': cf[2],
                        'context_head': context[0], 'context_rel': context[1], 'context_tail': context[2],
                        'cf_type': type, 'dataset': f'cfkgr_{size}'})

        dataset.append({'text': head_corruption_str,
                        'head': corruptions[0][0], 'rel': corruptions[0][1], 'tail': corruptions[0][2],
                        'rule': rule_id, 'id': example_id, 'type': 'head_corr',
                        'expected_label': 0, 'og_label': 0,
                        'cf_head': cf[0], 'cf_rel': cf[1], 'cf_tail': cf[2],
                        'context_head': context[0], 'context_rel': context[1], 'context_tail': context[2],
                        'cf_type': type, 'dataset': f'cfkgr_{size}'})

        dataset.append({'text': tail_corruption_str,
                        'head': corruptions[1][0], 'rel': corruptions[1][1], 'tail': corruptions[1][2],
                        'rule': rule_id, 'id': example_id, 'type': 'tail_corr',
                        'expected_label': 0, 'og_label': 0,
                        'cf_head': cf[0], 'cf_rel': cf[1], 'cf_tail': cf[2],
                        'context_head': context[0], 'context_rel': context[1], 'context_tail': context[2],
                        'cf_type': type, 'dataset': f'cfkgr_{size}'})

        dataset.append({'text': rel_corruption_str,
                        'head': corruptions[2][0], 'rel': corruptions[2][1], 'tail': corruptions[2][2],
                        'rule': rule_id, 'id': example_id, 'type': 'rel_corr',
                        'expected_label': 0, 'og_label': 0,
                        'cf_head': cf[0], 'cf_rel': cf[1], 'cf_tail': cf[2],
                        'context_head': context[0], 'context_rel': context[1], 'context_tail': context[2],
                        'cf_type': type, 'dataset': f'cfkgr_{size}'})

        for i, f in enumerate(near_fact_verb_str):
            dataset.append({'text': f,
                            'head': facts_to_retain_near[i][0], 'rel': facts_to_retain_near[i][1], 'tail': facts_to_retain_near[i][2],
                            'rule': rule_id, 'id': example_id, 'type': 'near_fact',
                            'expected_label': 1, 'og_label': 1,
                            'cf_head': cf[0], 'cf_rel': cf[1], 'cf_tail': cf[2],
                            'context_head': context[0], 'context_rel': context[1], 'context_tail': context[2],
                            'cf_type': type, 'dataset': f'cfkgr_{size}'})

        for i, f in enumerate(far_fact_verb_str):
            dataset.append({'text': f,
                            'head': facts_to_retain_far[i][0], 'rel': facts_to_retain_far[i][1], 'tail': facts_to_retain_far[i][2],
                            'rule': rule_id, 'id': example_id, 'type': 'far_fact',
                            'expected_label': 1, 'og_label': 1,
                            'cf_head': cf[0], 'cf_rel': cf[1], 'cf_tail': cf[2],
                            'context_head': context[0], 'context_rel': context[1], 'context_tail': context[2],
                            'cf_type': type, 'dataset': f'cfkgr_{size}'})
    else:
        raise NotImplementedError

    return dataset

def verbalize_entry(triple, cf, context, data, test_type):
    cf_verb = verbalize_triple(data, cf, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
    context_verb = verbalize_triple(data, context, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
    triple_verb = verbalize_triple(data, triple, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
    prompt = f'Hypothetical scenario: {cf_verb}\n\nContext: {context_verb}\n\n'
    if test_type in ["conclusion",
                        "head_corr", "rel_corr", "tail_corr",
                        "head_corr_far", "rel_corr_far", "tail_corr_far",
                        "head_corr_near", "rel_corr_near", "tail_corr_near"]:

        string = prompt + f'Is it then plausible that {triple_verb}, given that this is not the case in the real world?'
    elif test_type in ["far_fact", "near_fact"]:
        string = f'Is it then still plausible that {triple_verb}, given that this is the case in the real world?'
    else:
        raise ValueError('Test type invalid.')

    return string
