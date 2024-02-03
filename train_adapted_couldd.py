"""
Adapted from https://github.com/uma-pi1/kge to allow training solely on counterfactual triples
(Original code licensed under the MIT License)
"""

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from kge import Config, Dataset
from kge.job import Job, TrainingOrEvaluationJob
from kge.util import KgeLoss, KgeOptimizer, KgeSampler
from kge.job.train import _generate_worker_init_fn

SLOTS = [0, 1, 2]
S, P, O = SLOTS
SLOT_STR = ["s", "p", "o"]

class TrainingJob(TrainingOrEvaluationJob):
    """
    ### Adapted from LibKGE; original code can be found at https://github.com/uma-pi1/kge

    Abstract base job to train a single model with a fixed set of hyperparameters.

    Also used by jobs such as :class:`SearchJob`.

    Subclasses for specific training methods need to implement `_prepare` and
    `_process_batch`.

    """

    def __init__(
        self, config: Config, cf: torch.tensor, dataset: Dataset, epochs: int, lr: float, thresholds: dict, additional_samples: int = 0, additional_samples_type: str = 'relation', early_stop: bool = True, parent_job: Job = None, model=None
    ) -> None:

        super().__init__(config, dataset, parent_job)

        ################################## changes: some initializations COULDD-specific ###############################
        ### model
        self.model = model
        self.model.train() # set model into train mode

        ### data
        self.cf = cf
        self.thresholds = thresholds
        self.key = self.cf[1].item() if self.cf[1].item() in self.thresholds.keys() else -1
        self.train = pd.DataFrame(dataset.split('train').numpy(), columns=['head', 'relation', 'tail'])

        ### training details
        self.optimizer = KgeOptimizer.create(config, self.model)

        # adjust learning rate
        self.lr = lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        # create loss
        self.loss = KgeLoss.create(config)
        self.abort_on_nan: bool = config.get("train.abort_on_nan")
        self.device: str = self.config.get("job.device")
        self.epochs: int = epochs
        self.additional_samples = additional_samples
        self.additional_samples_type = additional_samples_type
        self.train_split = 'train'

        # early stop settings
        self.early_stop = early_stop
        self.early_stop_flag = False
        ################################################################################################################

        # trace settings
        self.config.check("train.trace_level", ["batch", "epoch"])
        self.trace_batch: bool = self.config.get("train.trace_level") == "batch"
        self.epoch: int = 0

        # attributes filled in by implementing classes
        self.loader = None
        self.num_examples = None
        self.type_str: Optional[str] = None

        if self.__class__ == TrainingJob:
            for f in Job.job_created_hooks:
                f(self)

    @staticmethod
    def create(
        config: Config, cf: torch.tensor, dataset: Dataset, epochs: int, lr: float, thresholds: dict, additional_samples: int, additional_samples_type: str, early_stop: bool, parent_job: Job = None, model=None
    ) -> "TrainingJob":
        """Factory method to create a training job."""
        if config.get("train.type") == "couldd":
            return TrainingJobCOULDD(config, cf, dataset, epochs, lr, thresholds, additional_samples, additional_samples_type, early_stop, parent_job, model=model)
        else:
            raise ValueError('Training type not implemented.')

    def _run(self) -> None:
        """Start/resume the training job and run to completion."""

        ########################## changes: add early stopping for COULDD ################################
        while self.epoch < self.epochs:
            if self.early_stop is True and self.early_stop_flag is True:
                # print('Early stop after epoch', self.epoch)
                break
        ################################################################################################

            # start a new epoch
            self.epoch += 1
            # self.config.log("Starting epoch {}...".format(self.epoch))
            trace_entry = self.run_epoch()
            # self.config.log("Finished epoch {}.".format(self.epoch))

            # update model metadata
            self.model.meta["train_job_trace_entry"] = self.trace_entry
            self.model.meta["train_epoch"] = self.epoch
            self.model.meta["train_config"] = self.config
            self.model.meta["train_trace_entry"] = trace_entry

        ##################################### end of changes ##############
            if self.epoch == self.epochs:
                # print('No early stop.')
                self.trace(event="train_completed")
        ###################################################################

    def save(self, filename) -> None:
        """Save current state to specified file"""
        self.config.log("Saving checkpoint to {}...".format(filename))
        checkpoint = self.save_to({})
        torch.save(
            checkpoint, filename,
        )

    def run_epoch(self) -> Dict[str, Any]:
        "Runs an epoch and returns its trace entry."

        # create initial trace entry
        self.current_trace["epoch"] = dict(
            type=self.type_str,
            scope="epoch",
            epoch=self.epoch,
            split=self.train_split,
            batches=len(self.loader),
            size=self.num_examples,
            lr=[group["lr"] for group in self.optimizer.param_groups],
        )

        # run pre-epoch hooks (may modify trace)
        for f in self.pre_epoch_hooks:
            f(self)

        # variables that record various statistics
        sum_loss = 0.0
        sum_penalty = 0.0
        sum_penalties = defaultdict(lambda: 0.0)
        epoch_time = -time.time()
        prepare_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer_time = 0.0

        # process each batch
        for batch_index, batch in enumerate(self.loader):
            # create initial batch trace (yet incomplete)
            self.current_trace["batch"] = {
                "type": self.type_str,
                "scope": "batch",
                "epoch": self.epoch,
                "split": self.train_split,
                "batch": batch_index,
                "batches": len(self.loader),
            }

            # run the pre-batch hooks (may update the trace)
            for f in self.pre_batch_hooks:
                f(self)

            # process batch (preprocessing + forward pass + backward pass on loss)
            done = False
            while not done:
                # try running the batch
                self.optimizer.zero_grad()
                batch_result: TrainingJob._ProcessBatchResult = self._process_batch(
                    batch_index, batch
                    )
                done = True
            sum_loss += batch_result.avg_loss * batch_result.size

            # determine penalty terms (forward pass)
            batch_forward_time = batch_result.forward_time - time.time()
            penalties_torch = self.model.penalty(
                epoch=self.epoch,
                batch_index=batch_index,
                num_batches=len(self.loader),
                batch=batch,
            )
            batch_forward_time += time.time()

            # backward pass on penalties
            batch_backward_time = batch_result.backward_time - time.time()
            penalty = 0.0
            for index, (penalty_key, penalty_value_torch) in enumerate(penalties_torch):
                penalty_value_torch.backward()
                penalty += penalty_value_torch.item()
                sum_penalties[penalty_key] += penalty_value_torch.item()
            sum_penalty += penalty
            batch_backward_time += time.time()

            cost_value = batch_result.avg_loss + penalty

            # abort on nan
            if self.abort_on_nan and math.isnan(cost_value):
                raise FloatingPointError("Cost became nan, aborting training job")

            # update parameters
            batch_optimizer_time = -time.time()
            self.optimizer.step()
            batch_optimizer_time += time.time()

            # update batch trace with the results
            self.current_trace["batch"].update(
                {
                    "size": batch_result.size,
                    "avg_loss": batch_result.avg_loss,
                    "penalties": [p.item() for k, p in penalties_torch],
                    "penalty": penalty,
                    "cost": cost_value,
                    "prepare_time": batch_result.prepare_time,
                    "forward_time": batch_forward_time,
                    "backward_time": batch_backward_time,
                    "optimizer_time": batch_optimizer_time,
                    "event": "batch_completed",
                }
            )

            # run the post-batch hooks (may modify the trace)
            for f in self.post_batch_hooks:
                f(self)

            # output, then clear trace
            if self.trace_batch:
                self.trace(**self.current_trace["batch"])
            self.current_trace["batch"] = None

            # update epoch times
            prepare_time += batch_result.prepare_time
            forward_time += batch_forward_time
            backward_time += batch_backward_time
            optimizer_time += batch_optimizer_time

        # all done; now trace and log
        epoch_time += time.time()
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back

        other_time = (
            epoch_time - prepare_time - forward_time - backward_time - optimizer_time
        )

        # add results to trace entry
        self.current_trace["epoch"].update(
            dict(
                avg_loss=sum_loss / self.num_examples,
                avg_penalty=sum_penalty / len(self.loader),
                avg_penalties={
                    k: p / len(self.loader) for k, p in sum_penalties.items()
                },
                avg_cost=sum_loss / self.num_examples + sum_penalty / len(self.loader),
                epoch_time=epoch_time,
                prepare_time=prepare_time,
                forward_time=forward_time,
                backward_time=backward_time,
                optimizer_time=optimizer_time,
                other_time=other_time,
                event="epoch_completed",
            )
        )

        # run hooks (may modify trace)
        for f in self.post_epoch_hooks:
            f(self)

        # output the trace, then clear it
        trace_entry = self.trace(
            **self.current_trace["epoch"], echo=False, echo_prefix="  ", log=True
        )
        self.current_trace["epoch"] = None

        ############addded: check whether new fact is integrated####################################################################
        self.model.eval()
        with torch.no_grad():
            if self.model.score_spo(*[self.cf[:, i].to(self.device) for i in range(3)], direction="o") >= self.thresholds[self.key]:
                self.early_stop_flag = True
        self.model.train()
        #############################################################################################################################

        return trace_entry

    def _prepare(self):
        """Prepare this job for running.

        Sets (at least) the `loader`, `num_examples`, and `type_str` attributes of this
        job to a data loader, number of examples per epoch, and a name for the trainer,
        repectively.

        Guaranteed to be called exactly once before running the first epoch.

        """
        super()._prepare()
        self.model.prepare_job(self)  # let the model add some hooks

    @dataclass
    class _ProcessBatchResult:
        """Result of running forward+backward pass on a batch."""

        avg_loss: float = 0.0
        size: int = 0
        prepare_time: float = 0.0
        forward_time: float = 0.0
        backward_time: float = 0.0

    def _process_batch(self, batch_index, batch) -> _ProcessBatchResult:
        "Breaks a batch into subbatches and processes them in turn."
        result = TrainingJob._ProcessBatchResult()
        self._prepare_batch(batch_index, batch, result)
        batch_size = result.size

        max_subbatch_size = (
            self._max_subbatch_size if self._max_subbatch_size > 0 else batch_size
        )
        for subbatch_start in range(0, batch_size, max_subbatch_size):
            # determine data used for this subbatch
            subbatch_end = min(subbatch_start + max_subbatch_size, batch_size)
            subbatch_slice = slice(subbatch_start, subbatch_end)
            self._process_subbatch(batch_index, batch, subbatch_slice, result)

        return result

    def _prepare_batch(self, batch_index, batch, result: _ProcessBatchResult):
        """Prepare the given batch for processing and determine the batch size.

        batch size must be written into result.size.
        """
        raise NotImplementedError

    def _process_subbatch(
        self, batch_index, batch, subbatch_slice, result: _ProcessBatchResult,
    ):
        """Run forward and backward pass on the given subbatch.

        Also update result.

        """
        raise NotImplementedError

class TrainingJobCOULDD(TrainingJob):
    """Samples SPO pairs and queries sp_ and _po, treating all other entities as negative."""

    def __init__(self, config, cf, dataset, epochs, lr, thresholds, additional_samples, additional_samples_type, early_stop, parent_job=None, model=None):
        super().__init__(config, cf, dataset, epochs, lr, thresholds, additional_samples, additional_samples_type, early_stop, parent_job, model)

        ########changes: define COULDD-specific parameters##########
        self.cf = self.cf.unsqueeze(0)
        self._sampler = KgeSampler.create(config, "negative_sampling", dataset)  # create negative sampler with original dataset

        self.type_str = "couldd"

        self.batch_size = self.additional_samples + 1
        self._max_subbatch_size: int = self.batch_size  # no subbatches

        if self.__class__ == TrainingJobCOULDD: # small change: COULDD instead of TrainingJobNegativeSampling
            for f in Job.job_created_hooks:
                f(self)
        #############################################################

    def _prepare(self):
        """Construct dataloader"""
        super()._prepare()

        self.num_examples = self.cf.shape[0] ### change: only sample counterfactual
        self.loader = torch.utils.data.DataLoader(
            range(self.num_examples),
            collate_fn=self._get_collate_fun(),
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.config.get("train.num_workers"),
            worker_init_fn=_generate_worker_init_fn(self.config),
            pin_memory=self.config.get("train.pin_memory"),
        )

    def _get_collate_fun(self):
        def collate(batch):
            ##########################changes: sample a batch as needed for COULDD#####################################
            triples = self.cf[batch, :].long().to(self.device)

            # sample additional samples if specified
            if self.additional_samples_type == 'relation':
                self.additional = self.train[self.train['relation'] == self.cf[0][1].item()]
            elif self.additional_samples_type == 'subgraph':
                self.additional = self.train[(self.train['head'].isin([self.cf[0][0].item(), self.cf[0][2].item()])) |
                                             (self.train['tail'].isin([self.cf[0][0].item(), self.cf[0][2].item()]))]
            elif self.additional_samples_type == 'uniform':
                self.additional = self.train
            else:
                raise ValueError("Additional sample type can be either 'relation', 'subgraph' or 'uniform'.")
            assert self.additional.shape[0] > 0
            self.additional.reset_index(drop=True, inplace=True)
            self.additional = torch.tensor(self.additional.values)
            idx = np.random.choice(self.additional.shape[0], min(self.additional_samples, self.additional.shape[0]), replace=False)
            triples = torch.cat([triples, self.additional[idx, :].to(self.device)])
            ############################################################################################################

            negative_samples = list()
            for slot in [S, P, O]:
                negative_samples.append(self._sampler.sample(triples.cpu(), slot)) # small change: added .cpu()
            return {"triples": triples, "negative_samples": negative_samples}

        return collate

    def _prepare_batch(
        self, batch_index, batch, result: TrainingJob._ProcessBatchResult
    ):
        # move triples and negatives to GPU. With some implementaiton effort, this may
        # be avoided.
        result.prepare_time -= time.time()
        batch["triples"] = batch["triples"].to(self.device)
        for ns in batch["negative_samples"]:
            ns.positive_triples = batch["triples"]
        batch["negative_samples"] = [
            ns.to(self.device) for ns in batch["negative_samples"]
        ]

        batch["labels"] = [None] * 3  # reuse label tensors b/w subbatches
        result.size = len(batch["triples"])
        result.prepare_time += time.time()

    def _process_subbatch(
        self,
        batch_index,
        batch,
        subbatch_slice,
        result: TrainingJob._ProcessBatchResult,
    ):
        # prepare
        result.prepare_time -= time.time()
        triples = batch["triples"][subbatch_slice]
        batch_negative_samples = batch["negative_samples"]
        batch_size = len(batch["triples"])
        subbatch_size = len(triples)
        result.prepare_time += time.time()
        labels = batch["labels"]  # reuse b/w subbatches

        # process the subbatch for each slot separately
        for slot in [S, P, O]:
            num_samples = self._sampler.num_samples[slot]
            if num_samples <= 0:
                continue

            # construct gold labels: first column corresponds to positives,
            # remaining columns to negatives
            if labels[slot] is None or labels[slot].shape != (
                subbatch_size,
                1 + num_samples,
            ):
                result.prepare_time -= time.time()
                labels[slot] = torch.zeros(
                    (subbatch_size, 1 + num_samples), device=self.device
                )
                labels[slot][:, 0] = 1
                result.prepare_time += time.time()

            # compute the scores
            result.forward_time -= time.time()
            scores = torch.empty((subbatch_size, num_samples + 1), device=self.device)
            scores[:, 0] = self.model.score_spo(
                triples[:, S], triples[:, P], triples[:, O], direction=SLOT_STR[slot],
            )
            result.forward_time += time.time()
            scores[:, 1:] = batch_negative_samples[slot].score(
                self.model, indexes=subbatch_slice
            )
            result.forward_time += batch_negative_samples[slot].forward_time
            result.prepare_time += batch_negative_samples[slot].prepare_time

            # compute loss for slot in subbatch (concluding the forward pass)
            result.forward_time -= time.time()
            loss_value_torch = (
                self.loss(scores, labels[slot], num_negatives=num_samples) / batch_size
            )
            result.avg_loss += loss_value_torch.item()
            result.forward_time += time.time()

            # backward pass for this slot in the subbatch
            result.backward_time -= time.time()
            loss_value_torch.backward()
            result.backward_time += time.time()
