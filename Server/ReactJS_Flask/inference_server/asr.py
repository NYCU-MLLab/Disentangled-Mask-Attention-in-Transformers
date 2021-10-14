

import os
import torch
import speechbrain as sb
import logging
from speechbrain.nnet.attention import DisentangledMaskAttention

class ASR(sb.Brain):
    @torch.no_grad()
    def compute_forward(self, wavs, wav_lens):
        """Forward computations from the waveform batches to the output probabilities."""
        wavs = wavs.to(self.device)
        wav_lens = wav_lens.to(self.device)

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.hparams.CNN(feats)
        enc_out = self.hparams.Transformer.encode(src)

        # Compute outputs
        hyps, _ = self.hparams.test_search(enc_out, wav_lens)

        predicted_sentence = [
            self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
        ]
        predicted_sentence = ["".join(p) for p in predicted_sentence]

        predicted_tokens = [self.tokenizer.decode_ids([idx]) for idx in hyps[0]] + ["<eos>"]

        return predicted_tokens, predicted_sentence
