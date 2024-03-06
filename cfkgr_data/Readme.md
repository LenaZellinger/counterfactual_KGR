This folder contains the datasets used in our experiments.
Below is an overview over the columns in the data files.

**"head"**, **"rel"**, **"tail"**: head, relation and tail of the instance that should be scored expressed via Wikidata labels

**"head_libkge"**, **"rel_libkge"**, **"tail_libkge"**: head, relation and tail of the instance that should be scored expressed via LibKGE ids (this expression is compatible with COULDD and the CoDEx models)

**"type"**: type of the test case; possible entries are: conclusion, far_fact, near_fact, head_corr, head_corr_near, head_corr_far, rel_corr, rel_corr_near, rel_corr_far, tail_corr, tail_corr_near, tail_corr_far

**"expected_label"**: label that the test instance has in the counterfactual graph

**"original_label"**: label that the test instance has in the original graph

**"cf_head"**, **"cf_rel"**, **"cf_tail"**: head, relation and tail of the current hypothetical scenario expressed via Wikidata labels

**"cf_head_libkge"**, **"cf_rel_libkge"**, **"cf_tail_libkge"**: head, relation and tail of the current hypothetical scenario expressed via LibKGE ids

**"context_head"**, **"context_rel"**, **"context_tail"**: head, relation and tail of the context triggering the rule expressed via Wikidata labels

**"context_head_libkge"**, **"context_rel_libkge"**, **"context_tail_libkge"**: head, relation and tail of the context triggering the rule expressed via LibKGE ids

**"rule"**: id of rule that produced the test instance

**"id"**: id of the test instance; entries with the same id are test cases associated with the same hypothetical scenario

**"cf_type"**: 0 if hypothetical scenario was created from the first rule atom; 1 if it was generated from second rule atom

**"dataset"**: dataset that this instance belongs to

**"cf"**, **"context"**, **"test_instance"**: full triples of hypothetical scenario, context and current test instance expressed via Wikidata labels

**"cf_libkge"**, **"context_libkge"**, **"test_instance_libkge"**: full triples of hypothetical scenario, context and current test instance expressed via libkge ids

**"text"**: verbalization of the test instance
