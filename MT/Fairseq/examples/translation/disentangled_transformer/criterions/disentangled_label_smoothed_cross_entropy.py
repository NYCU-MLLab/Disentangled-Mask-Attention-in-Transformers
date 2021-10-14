# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch.nn import functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from sklearn.mixture import GaussianMixture

@dataclass
class DisentangledLabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def cluster_loss(net_output):
    cluster_loss = net_output[1]["enc_cluster_loss"].float().mean((1, 2)).sum() +\
                   net_output[1]["dec_cluster_loss"].float().mean((1, 2)).sum() +\
                   net_output[1]["dec_enc_cluster_loss"].float().mean((1, 2)).sum()

    return cluster_loss

def mi_loss(net_output):
    mi_loss = net_output[1]["enc_mi_loss"].float().mean((1, 2)).sum() +\
              net_output[1]["dec_mi_loss"].float().mean((1, 2)).sum() +\
              net_output[1]["dec_enc_mi_loss"].float().mean((1, 2)).sum()

    return mi_loss

def cluster_div_loss(net_output):
    cluster_div_loss = net_output[1]["enc_cluster_div_loss"].float().mean((1, 2)).sum() +\
                       net_output[1]["dec_cluster_div_loss"].float().mean((1, 2)).sum() +\
                       net_output[1]["dec_enc_cluster_div_loss"].float().mean((1, 2)).sum()

    return cluster_div_loss

@register_criterion(
    "disentangled_label_smoothed_cross_entropy", dataclass=DisentangledLabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

        self.src_unk_index = task.src_dict.unk_index
        self.tgt_unk_index = task.tgt_dict.unk_index

        self.data = {}

        self.initialization_tokens = 50001
        self.max_initialization_tokens = 50000
        self.initialization_data = {
            "enc_hidden_states": [],
            "dec_hidden_states": [],
            "dec_enc_hidden_states": [],
            "enc_query": [],
            "dec_query": [],
            "dec_enc_query": []
        }
        self.reinitialized = True

    def gmm_pretraining(self, embeddings: torch.Tensor, clusters: int, mu: torch.Tensor, log_cov: torch.Tensor, log_prior: torch.Tensor):
        embeddings = embeddings.numpy()

        gmm = GaussianMixture(n_components=clusters, random_state=0, covariance_type="diag", verbose=True).fit(embeddings)

        mu.data.copy_(torch.from_numpy(gmm.means_))
        log_cov.data.copy_(torch.from_numpy(gmm.covariances_))
        log_prior.data.copy_(torch.from_numpy(gmm.weights_))

    def initialize_model(self, model, initialize_disentangled_head=False):
        for k, v in self.initialization_data.items():
            self.initialization_data[k] = torch.cat(v, dim=0)

        H = self.initialization_data["enc_hidden_states"].size()[1]

        for layer_idx, layer in enumerate(model.encoder.layers):
            for h in range(H):
                print("Pretraining encoder GMM parameters ... (layer={}, head={})".format(layer_idx, h))
                self.gmm_pretraining(
                    self.initialization_data["enc_hidden_states"][:self.max_initialization_tokens, h, layer_idx].cpu(),
                    model.args.clusters,
                    layer.self_attn.attn.tok_mu[h:h+1],
                    layer.self_attn.attn.tok_log_var[h:h+1],
                    layer.self_attn.attn.tok_log_prior[h:h+1]
                )

            if initialize_disentangled_head:
                print("Pretraining encoder GMM parameters ... (layer={})".format(layer_idx))
                self.gmm_pretraining(
                    self.initialization_data["enc_query"][:self.max_initialization_tokens*H, layer_idx].cpu(),
                    model.args.clusters,
                    layer.self_attn.attn.head_mu,
                    layer.self_attn.attn.head_log_var,
                    layer.self_attn.attn.head_log_prior
                )

        for layer_idx, layer in enumerate(model.decoder.layers):
            for h in range(H):
                print("Pretraining decoder GMM parameters ... (layer={}, head={})".format(layer_idx, h))
                self.gmm_pretraining(
                    self.initialization_data["dec_hidden_states"][:self.max_initialization_tokens, h, layer_idx].cpu(),
                    model.args.clusters,
                    layer.self_attn.attn.tok_mu[h:h+1],
                    layer.self_attn.attn.tok_log_var[h:h+1],
                    layer.self_attn.attn.tok_log_prior[h:h+1]
                )

                print("Pretraining decoder-encoder GMM parameters ... (layer={}, head={})".format(layer_idx, h))
                self.gmm_pretraining(
                    self.initialization_data["dec_enc_hidden_states"][:self.max_initialization_tokens, h, layer_idx].cpu(),
                    model.args.clusters,
                    layer.encoder_attn.attn.tok_mu[h:h+1],
                    layer.encoder_attn.attn.tok_log_var[h:h+1],
                    layer.encoder_attn.attn.tok_log_prior[h:h+1]
                )

            if initialize_disentangled_head:
                print("Pretraining decoder GMM parameters ... (layer={})".format(layer_idx))
                self.gmm_pretraining(
                    self.initialization_data["dec_query"][:self.max_initialization_tokens*H, layer_idx].cpu(),
                    model.args.clusters,
                    layer.self_attn.attn.head_mu,
                    layer.self_attn.attn.head_log_var,
                    layer.self_attn.attn.head_log_prior
                )

                print("Pretraining decoder-encoder  GMM parameters ... (layer={})".format(layer_idx))
                self.gmm_pretraining(
                    self.initialization_data["dec_enc_query"][:self.max_initialization_tokens*H, layer_idx].cpu(),
                    model.args.clusters,
                    layer.encoder_attn.attn.head_mu,
                    layer.encoder_attn.attn.head_log_var,
                    layer.encoder_attn.attn.head_log_prior
                )


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        num_updates = model.num_updates
        unk_augmentation_rate = model.args.unk_augmentation_rate

        with torch.no_grad():
            if model.training and unk_augmentation_rate > 0:
                src_tokens = sample["net_input"]["src_tokens"]
                prev_output_tokens = sample["net_input"]["prev_output_tokens"]

                src_augmented_mask = torch.bernoulli(
                    torch.full(src_tokens.shape, unk_augmentation_rate)
                ).to(src_tokens.device).bool()
                tgt_augmented_mask = torch.bernoulli(
                    torch.full(prev_output_tokens.shape, unk_augmentation_rate)
                ).to(prev_output_tokens.device).bool()

                src_tokens[src_augmented_mask] = self.src_unk_index
                prev_output_tokens[:, 1:][tgt_augmented_mask[:, 1:]] = self.tgt_unk_index

                sample["net_input"]["src_tokens"] = src_tokens
                sample["net_input"]["prev_output_tokens"] = prev_output_tokens

        regular_attn = model.args.regular_attn
        mi_loss_weight = model.args.mi_loss_weight
        cluster_loss_weight = model.args.cluster_loss_weight

        if num_updates >= 5000 and not self.reinitialized:
            self.initialization_tokens = 0
            self.reinitialized = True

            self.initialization_data = {
                "enc_hidden_states": [],
                "dec_hidden_states": [],
                "dec_enc_hidden_states": [],
                "enc_query": [],
                "dec_query": [],
                "dec_enc_query": []
            }

        if self.initialization_tokens < self.max_initialization_tokens:
            with torch.no_grad():
                print("Collecting embeddings ... {} ...".format(self.initialization_tokens))
                net_output = model(**sample["net_input"], regular_attn=regular_attn, debug=True)

                for prefix in ["enc_", "dec_", "dec_enc_"]:
                    B, H, _, L, D = net_output[1][prefix+"_query"].size()

                    self.initialization_data[prefix+"hidden_states"].append(
                        net_output[1][prefix+"_hidden_states"].permute(0, 3, 1, 2, 4).reshape(B*L, H, -1, D)
                    )
                    self.initialization_data[prefix+"query"].append(
                        net_output[1][prefix+"_query"].permute(0, 3, 1, 2, 4).reshape(B*L*H, -1, D)
                    )

                self.initialization_tokens += B*L

                if self.initialization_tokens >= self.max_initialization_tokens:
                    self.initialize_model(model, initialize_disentangled_head=self.reinitialized)

            fake_tensor_need_grad = torch.tensor([0.], requires_grad=True).to(net_output[0])
        else:
            net_output = model(**sample["net_input"], regular_attn=regular_attn)
            fake_tensor_need_grad = 0

        loss, nll_loss, cluster_loss, cluster_div_loss, mi_loss = self.compute_loss(
            model, net_output, sample, mi_loss_weight, cluster_loss_weight, regular_attn, reduce=reduce
        )
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "cluster_loss": cluster_loss.data,
            "cluster_div_loss": cluster_div_loss.data,
            "mi_loss": mi_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        loss = loss + fake_tensor_need_grad
        
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, mi_loss_weight, cluster_loss_weight,
                     regular_attn=True, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )

        if not regular_attn:
            _cluster_loss = 0.01*cluster_loss(net_output)
            _cluster_div_loss = cluster_div_loss(net_output)
            _mi_loss = mi_loss(net_output)
        else:
            _cluster_loss = _cluster_div_loss = _mi_loss =torch.zeros((1,)).to(net_output[0].device)

        loss = loss + _cluster_loss + cluster_loss_weight*_cluster_div_loss + mi_loss_weight*_mi_loss

        return loss, nll_loss, _cluster_loss, _cluster_div_loss, _mi_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        cluster_loss_sum = sum(log.get("cluster_loss", 0) for log in logging_outputs)
        cluster_div_loss_sum = sum(log.get("cluster_div_loss", 0) for log in logging_outputs)
        mi_loss_sum = sum(log.get("mi_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "cluster_loss", cluster_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "cluster_div_loss", cluster_div_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "mi_loss", mi_loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
