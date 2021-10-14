#!/usr/bin/env/python3
"""Recipe for training a Transducer ASR system with librispeech.
The system employs an encoder, a decoder, and an joint network
between them. Decoding is performed with beamsearch coupled with a neural
language model.

To run this recipe, do the following:
> python train.py hparams/train.yaml

With the default hyperparameters, the system employs a CRDNN encoder.
The decoder is based on a standard  GRU. Beamsearch coupled with a RNN
language model is used on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
LibriSpeech dataset (960 h).

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.


Authors
 * Abdel Heba 2020
 * Mirco Ravanelli 2020
 * Ju-Chieh Chou 2020
 * Peter Plantinga 2020
"""

import os
import sys
import torch
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path

logger = logging.getLogger(__name__)

# Define training procedure


class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_with_bos, token_with_bos_lens = batch.tokens_bos
        # wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, wav_lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                wav_lens = torch.cat([wav_lens, wav_lens])
                batch.sig = wavs, wav_lens
                tokens_with_bos = torch.cat(
                    [tokens_with_bos, tokens_with_bos], dim=0
                )
                token_with_bos_lens = torch.cat(
                    [token_with_bos_lens, token_with_bos_lens]
                )
                batch.tokens_bos = tokens_with_bos, token_with_bos_lens
            if hasattr(self.modules, "augmentation"):
                wavs = self.modules.augmentation(wavs, wav_lens)

        # Forward pass
        feats = self.hparams.compute_features(wavs)
        feats = self.modules.normalize(feats, wav_lens)
        x = self.modules.enc(feats.detach())
        e_in = self.modules.emb(tokens_with_bos)
        h, _ = self.modules.dec(e_in)
        # Joint network
        # add labelseq_dim to the encoder tensor: [B,T,H_enc] => [B,T,1,H_enc]
        # add timeseq_dim to the decoder tensor: [B,U,H_dec] => [B,1,U,H_dec]
        joint = self.modules.Tjoint(x.unsqueeze(2), h.unsqueeze(1))

        # Output layer for transducer log-probabilities
        logits = self.modules.transducer_lin(joint)
        p_transducer = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.TRAIN:
            return_CTC = False
            return_CE = False
            current_epoch = self.hparams.epoch_counter.current
            if (
                hasattr(self.hparams, "ctc_cost")
                and current_epoch <= self.hparams.number_of_ctc_epochs
            ):
                return_CTC = True
                # Output layer for ctc log-probabilities
                out_ctc = self.modules.enc_lin(x)
                p_ctc = self.hparams.log_softmax(out_ctc)
            if (
                hasattr(self.hparams, "ce_cost")
                and current_epoch <= self.hparams.number_of_ce_epochs
            ):
                return_CE = True
                # Output layer for ctc log-probabilities
                p_ce = self.modules.dec_lin(h)
                p_ce = self.hparams.log_softmax(p_ce)
            if return_CE and return_CTC:
                return p_ctc, p_ce, p_transducer, wav_lens
            elif return_CTC:
                return p_ctc, p_transducer, wav_lens
            elif return_CE:
                return p_ce, p_transducer, wav_lens
            else:
                return p_transducer, wav_lens

        elif stage == sb.Stage.VALID:
            best_hyps, scores, _, _ = self.hparams.Greedysearcher(x)
            return p_transducer, wav_lens, best_hyps
        else:
            (
                best_hyps,
                best_scores,
                nbest_hyps,
                nbest_scores,
            ) = self.hparams.Beamsearcher(x)
            return p_transducer, wav_lens, best_hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (Transducer+(CTC+NLL)) given predictions and targets."""

        ids = batch.id
        current_epoch = self.hparams.epoch_counter.current
        tokens, token_lens = batch.tokens
        tokens_eos, token_eos_lens = batch.tokens_eos
        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            token_eos_lens = torch.cat([token_eos_lens, token_eos_lens], dim=0)
            tokens = torch.cat([tokens, tokens], dim=0)
            token_lens = torch.cat([token_lens, token_lens], dim=0)

        if stage == sb.Stage.TRAIN:
            if len(predictions) == 4:
                p_ctc, p_ce, p_transducer, wav_lens = predictions
                CTC_loss = self.hparams.ctc_cost(
                    p_ctc, tokens, wav_lens, token_lens
                )
                CE_loss = self.hparams.ce_cost(
                    p_ce, tokens_eos, length=token_eos_lens
                )
                tokens = tokens.long()
                loss_transducer = self.hparams.transducer_cost(
                    p_transducer, tokens, wav_lens, token_lens
                )
                loss = (
                    self.hparams.ctc_weight * CTC_loss
                    + self.hparams.ce_weight * CE_loss
                    + (1 - (self.hparams.ctc_weight + self.hparams.ce_weight))
                    * loss_transducer
                )
            elif len(predictions) == 3:
                # one of the 2 heads (CTC or CE) is still computed
                # CTC alive
                if current_epoch <= self.hparams.number_of_ctc_epochs:
                    p_ctc, p_transducer, wav_lens = predictions
                    CTC_loss = self.hparams.ctc_cost(
                        p_ctc, tokens, wav_lens, token_lens
                    )
                    tokens = tokens.long()
                    loss_transducer = self.hparams.transducer_cost(
                        p_transducer, tokens, wav_lens, token_lens
                    )
                    loss = (
                        self.hparams.ctc_weight * CTC_loss
                        + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
                # CE for decoder alive
                else:
                    p_ce, p_transducer, wav_lens = predictions
                    CE_loss = self.hparams.ce_cost(
                        p_ce, tokens_eos, length=token_eos_lens
                    )
                    tokens = tokens.long()
                    loss_transducer = self.hparams.transducer_cost(
                        p_transducer, tokens, wav_lens, token_lens
                    )
                    loss = (
                        self.hparams.ce_weight * CE_loss
                        + (1 - self.hparams.ctc_weight) * loss_transducer
                    )
            else:
                p_transducer, wav_lens = predictions
                tokens = tokens.long()
                loss = self.hparams.transducer_cost(
                    p_transducer, tokens, wav_lens, token_lens
                )
        else:
            p_transducer, wav_lens, predicted_tokens = predictions
            tokens = tokens.long()
            loss = self.hparams.transducer_cost(
                p_transducer, tokens, wav_lens, token_lens
            )

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                self.tokenizer.decode_ids(utt_seq).split(" ")
                for utt_seq in predicted_tokens
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # Defining tokenizer and loading it
    # To avoid mismatch, we have to use the same tokenizer used for LM training
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["blank_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["blank_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_datasets


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # 1.  # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["data_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": hparams["train_csv"],
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets = dataio_prepare(hparams)

    # We download the pretrained LM and the tokenizer from HuggingFace (or elsewhere
    # depending on the path given in the YAML file). The tokenizer is loaded at
    # the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamicaly add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = hparams["tokenizer"]

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], test_loader_kwargs=hparams["test_dataloader_opts"]
        )
