"""This file verbalizes a given statement."""
from data_generation.generate_M.verbalizations import rel_map_context, reverse_rel_map_context, verbalize_triple  # use same verbalizations as for M

def verbalize_entry(entry,
                    dataset,
                    cf,
                    context,
                    type):
    # verbalize triples for annotation/gpt experiments
    cf_verb = verbalize_triple(dataset, cf, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
    context_verb = verbalize_triple(dataset, context, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)
    conclusion_verb = verbalize_triple(dataset, entry, rel_mapping=rel_map_context, reverse_rel_mapping=reverse_rel_map_context)

    prompt = f'Hypothetical scenario: {cf_verb}\n\nContext: {context_verb}\n\n'

    # original label 0
    if type in ['conclusion',
                'head_corr', 'rel_corr', 'tail_corr',
                'head_corr_far', 'rel_corr_far', 'tail_corr_far',
                'head_corr_near', 'rel_corr_near', 'tail_corr_near']:

        conclusion_str = prompt + f'Is it then plausible that {conclusion_verb}, given that this is not the case in the real world?'

    # original label 1
    elif type in ['near_fact', 'far_fact']:
        conclusion_str = prompt + f'Is it then still plausible that {conclusion_verb}, given that this is the case in the real world?'
    else:
        raise ValueError('Unknown type.')

    return conclusion_str
