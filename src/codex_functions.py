########################################################################################################################
# This file contains functions from https://github.com/tsafavi/codex/blob/master/scripts/tc.py
# Changes to the functions are indicated via comments

# The credit for all following functions goes to https://github.com/tsafavi/codex
########################################################################################################################

import os

import pandas as pd
import torch

import kge.model
import kge.util.sampler
from kge import Config
from src.utils import set_seed


def get_X_y(model, pos_spo, neg_spo=None):
    """
    :param model: kge.model.KgeModel
    :param pos_spo: torch.Tensor of positive triples
    :param neg_spo: torch.Tensor of negative triples
    :return X: torch.Tensor of [pos_scores, neg_scores]
    :return y: torch.Tensor of [1s, 0s]
    """
    ### Change: added ###
    model.eval()
    #####################
    pos_scores = model.score_spo(*[pos_spo[:, i] for i in range(3)], direction="o")
    ### Change: allow to only use positive triples ###
    if neg_spo is not None:
        neg_scores = model.score_spo(*[neg_spo[:, i] for i in range(3)], direction="o")

        X = torch.reshape(torch.cat((pos_scores, neg_scores)), (-1, 1))
        y = torch.cat(
            (
                torch.ones_like(pos_scores, device="cpu"),
                torch.zeros_like(neg_scores, device="cpu"),
            )
        )
    else:
        X = torch.reshape(pos_scores, (-1, 1))
        y = torch.ones_like(pos_scores, device=X.device)
    ##################################################

    return X, y


def generate_neg_spo(dataset, split, negative_type="uniform", num_samples=1):
    """
    :param dataset: kge.dataset.Dataset
    :param split: one of "valid", "test"
    :param negative_type: one of "uniform", "frequency"
    :param num_samples: number of negatives per positive
    :return: torch.Tensor of randomly generated negative triples
    """
    ### Change: added seed for reproducibility ####
    set_seed(0)
    ###############################################
    # Sample corrupted object entities
    if negative_type == "uniform":
        sampler = kge.util.sampler.KgeUniformSampler(
            Config(), "negative_sampling", dataset
        )
    elif negative_type == "frequency":
        sampler = kge.util.sampler.KgeFrequencySampler(
            Config(), "negative_sampling", dataset
        )
    else:
        raise ValueError(f"Negative sampling type {negative_type} not recognized")

    print(
        "Generating",
        num_samples,
        "negatives per positive with",
        negative_type,
        "sampling on the",
        split,
        "split",
    )

    spo = dataset.split(split)
    neg_o = sampler._sample(spo, 2, num_samples=num_samples)

    neg_spo = torch.cat(
        (
            torch.repeat_interleave(spo[:, :2].long(), num_samples, dim=0),
            torch.reshape(neg_o, (-1, 1)),
        ),
        dim=1,
    )

    return neg_spo


def load_neg_spo(dataset, size="s"):
    """
    :param dataset: kge.dataset.Dataset
    :return: torch.Tensor of negative triples loaded from directory
    """
    negs = []
    ### Change: different path structure ###
    path_to_codex_folder = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
    for split in ("valid_negatives", "test_negatives"):
        triples = pd.read_csv(
            os.path.join(path_to_codex_folder, 'data', 'triples', "codex-" + size, split + ".txt"),
            sep="\t",
            header=None,
        ).values
    #########################################

        # Convert string IDs to integer IDs
        entity_ids = dict(map(reversed, enumerate(dataset.entity_ids())))
        relation_ids = dict(map(reversed, enumerate(dataset.relation_ids())))

        triples = [
            [entity_ids[s], relation_ids[p], entity_ids[o]] for (s, p, o) in triples
        ]

        negs.append(torch.tensor(triples, dtype=torch.long, device="cpu"))

    return negs

# no changes in the following function
def get_threshold(scores, labels):
    """
    :param scores: torch.tensor of prediction scores
    :param labels: torch.tensor of triple labels
    :return threshold: best decision threshold for these scores
    """
    predictions = ((scores.view(-1, 1) >= scores.view(1, -1)).long()).t()

    accuracies = (predictions == labels.view(-1)).float().sum(dim=1)
    accuracies_max = accuracies.max()

    threshold = scores[accuracies_max == accuracies].min().item()
    return threshold

