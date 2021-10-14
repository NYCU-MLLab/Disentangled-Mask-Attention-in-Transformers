"""Decoders and output normalization for Sync-Transformer.

Author:
    Andy Huang 2021
    Other authors in ./transducer.py
"""
import torch
from speechbrain.lobes.models.sync_transformer.TransformerASR import SyncTransformerASR
from torch.nn.utils.rnn import pad_sequence


class StreamingBeamSearcher(torch.nn.Module):
    """
    This class implements the beam-search algorithm for the Streaming model.

    Parameters
    ----------
    decode_network_lst : list
        List of prediction network (PN) layers.
    tjoint: transducer_joint module
        This module perform the joint between TN and PN.
    classifier_network : list
        List of output layers (after performing joint between TN and PN)
        exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob
    blank_id : int
        The blank symbol/index.
    beam : int
        The width of beam. Greedy Search is used when beam = 1.
    nbest : int
        Number of hypotheses to keep.
    lm_module : torch.nn.ModuleList
        Neural networks modules for LM.
    lm_weight : float
        The weight of LM when performing beam search (λ).
        log P(y|x) + λ log P_LM(y). (default: 0.3)
    state_beam : float
        The threshold coefficient in log space to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (beam_hyps), if not, end the while loop.
        Reference: https://arxiv.org/pdf/1911.01629.pdf
    expand_beam : float
        The threshold coefficient to limit the number of expanded hypotheses
        that are added in A (process_hyp).
        Reference: https://arxiv.org/pdf/1911.01629.pdf
        Reference: https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/simple-decoder.cc (See PruneToks)

    Example
    -------
    searcher = TransducerBeamSearcher(
        decode_network_lst=[hparams["emb"], hparams["dec"]],
        tjoint=hparams["Tjoint"],
        classifier_network=[hparams["transducer_lin"]],
        blank_id=0,
        beam_size=hparams["beam_size"],
        nbest=hparams["nbest"],
        lm_module=hparams["lm_model"],
        lm_weight=hparams["lm_weight"],
        state_beam=2.3,
        expand_beam=2.3,
    )
    >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
    >>> import speechbrain as sb
    >>> emb = sb.nnet.embedding.Embedding(
    ...     num_embeddings=35,
    ...     embedding_dim=3,
    ...     consider_as_one_hot=True,
    ...     blank_id=0
    ... )
    >>> dec = sb.nnet.RNN.GRU(
    ...     hidden_size=10, input_shape=(1, 40, 34), bidirectional=False
    ... )
    >>> lin = sb.nnet.linear.Linear(input_shape=(1, 40, 10), n_neurons=35)
    >>> joint_network= sb.nnet.linear.Linear(input_shape=(1, 1, 40, 35), n_neurons=35)
    >>> tjoint = Transducer_joint(joint_network, joint="sum")
    >>> searcher = TransducerBeamSearcher(
    ...     decode_network_lst=[emb, dec],
    ...     tjoint=tjoint,
    ...     classifier_network=[lin],
    ...     blank_id=0,
    ...     beam_size=1,
    ...     nbest=1,
    ...     lm_module=None,
    ...     lm_weight=0.0,
    ... )
    >>> enc = torch.rand([1, 20, 10])
    >>> hyps, scores, _, _ = searcher(enc)
    """

    def __init__(
        self,
        model: SyncTransformerASR,
        classifier,
        blank_id,
        beam_size=4,
        nbest=5,
        lm_module=None,
        lm_weight=0.0,
        state_beam=2.3,
        expand_beam=2.3,
        max_steps=10
    ):
        super(StreamingBeamSearcher, self).__init__()
        self.model = model
        self.classifier = classifier

        assert isinstance(model, SyncTransformerASR), "Only sync-Transformer is supported."

        self.blank_id = blank_id
        self.beam_size = beam_size
        self.nbest = nbest
        self.lm = lm_module
        self.lm_weight = lm_weight
        self.max_steps = max_steps

        if lm_module is None and lm_weight > 0:
            raise ValueError("Language model is not provided.")

        self.state_beam = state_beam
        self.expand_beam = expand_beam
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if self.beam_size <= 1:
            self.searcher = self.transducer_greedy_decode
        else:
            self.searcher = self.transducer_beam_search_decode

    def forward(self, tn_output):
        """
        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        Topk hypotheses
        """

        hyps = self.searcher(tn_output)
        return hyps

    @torch.no_grad()   
    def transducer_greedy_decode(self, tn_output):        
        """Transducer greedy decoder is a greedy decoder over batch which apply Transducer rules:
            1- for each time step in the Transcription Network (TN, encoder) output:
                -> Update the ith utterance only if
                    the previous target != the new one (we save the hiddens and the target)
                -> otherwise:
                ---> keep the previous target prediction from the decoder

        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """
        assert "Unsupport yet !!!"

        hyp = {
            "prediction": [[] for _ in range(tn_output.size(0))],
            "logp_scores": [0.0 for _ in range(tn_output.size(0))],
        }
        # prepare BOS = Blank for the Prediction Network (PN)
        input_PN = [
            torch.ones(
                (1, 1),
                device=tn_output.device,
                dtype=torch.int32,
            )
            * self.blank_id
            for _ in range(tn_output.size(0))
        ]
        # First forward-pass on PN
        # For each time step
        for t_step in range(tn_output.size(1)):
            # Batch hidden update
            have_update_hyp = [i for i in range (tn_output.size(0))]
            steps = 0

            while True:
                # Select sentence to update
                # And do a forward steps + generated hidden
                selected_input_PN = self._get_sentence_to_update(
                    have_update_hyp, input_PN
                )
                selected_input_last_idx = torch.as_tensor([input_PN[i].size(-1)-1 for i in have_update_hyp]).long()
                selected_input_last_idx = selected_input_last_idx.view(-1, 1, 1)
                
                out_PN = self._forward_PN(
                    selected_input_PN, tn_output[have_update_hyp, t_step:t_step+1]
                )
                out_PN = self.classifier(out_PN)

                selected_input_last_idx = selected_input_last_idx.expand(-1, -1, out_PN.size(-1))
                logits = out_PN[:, 0].gather(dim=1, index=selected_input_last_idx.to(out_PN.device)).squeeze(1)

                # Sort outputs at time
                logp_targets, positions = torch.max(self.softmax(logits), dim=1)

                _have_update_hyp = []
                for org_i, i in zip(have_update_hyp, range(positions.size(0))):
                    # Update hiddens only if
                    # 1- current prediction is non blank
                    if positions[i].item() != self.blank_id:
                        hyp["prediction"][org_i].append(positions[i].item())
                        hyp["logp_scores"][org_i] += logp_targets[i]
                        input_PN[org_i] = torch.cat([input_PN[org_i], positions[None, i:i+1]], dim=1)
                        if input_PN[org_i].size()[1] < 1000:
                            _have_update_hyp.append(org_i)

                have_update_hyp = _have_update_hyp
                steps += 1
                if len(have_update_hyp) == 0 or steps >= self.max_steps:
                    break

        return (
            hyp["prediction"],
            torch.Tensor(hyp["logp_scores"]).exp().mean(),
            None,
            None,
        )

    @torch.no_grad()    
    def transducer_beam_search_decode(self, tn_output):
        """Streaming beam search decoder is a beam search decoder over batch which apply Transducer rules:
            1- for each utterance:
                2- for each time steps in the Transcription Network (TN, encoder) output:
                    -> Do forward on PN (decoder)
                    -> Select topK <= beam
                    -> Do a while loop extending the hyps until we reach blank
                        -> otherwise:
                        --> extend hyp by the new token

        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """

        # min between beam and max_target_lent
        nbest_batch = []
        nbest_batch_score = []
        for i_batch in range(tn_output.size(0)):
            # if we use RNN LM keep there hiddens
            # prepare BOS = Blank for the Prediction Network (PN)
            # Prepare Blank prediction
            blank = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.blank_id
            )
            input_PN = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.blank_id
            )
            # First forward-pass on PN
            hyp = {
                "prediction": [self.blank_id],
                "logp_score": 0.0,
                "hidden_dec": None,
            }
            if self.lm_weight > 0:
                lm_dict = {"hidden_lm": None}
                hyp.update(lm_dict)
            beam_hyps = [hyp]

            # For each time step
            for t_step in range(tn_output.size(1)):
                # get hyps for extension
                process_hyps = beam_hyps
                beam_hyps = []
                steps = 1

                while True:
                    if len(beam_hyps) >= self.beam_size:
                        break
                    if steps >= self.max_steps:
                        beam_hyps = process_hyps
                        break
                    steps += 1

                    # Add norm score
                    a_best_hyp = max(
                        process_hyps,
                        key=lambda x: x["logp_score"] / len(x["prediction"]),
                    )

                    # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                    if len(beam_hyps) > 0:
                        b_best_hyp = max(
                            beam_hyps,
                            key=lambda x: x["logp_score"]
                            / len(x["prediction"]),
                        )
                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]
                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    # remove best hyp from process_hyps
                    process_hyps.remove(a_best_hyp)

                    # forward PN
                    input_PN = torch.as_tensor(a_best_hyp["prediction"], device=tn_output.device, dtype=torch.int32)
                    out_PN = self._forward_PN(
                        input_PN.view(1, -1),
                        tn_output[i_batch:i_batch+1, t_step:t_step+1],
                    )
                    log_probs = self.classifier(out_PN[0, 0, -1]).log_softmax(dim=-1)

                    if self.lm_weight > 0:
                        log_probs_lm, hidden_lm = self._lm_forward_step(
                            input_PN, a_best_hyp["hidden_lm"]
                        )

                    # Sort outputs at time
                    logp_targets, positions = torch.topk(
                        log_probs.view(-1), k=self.beam_size, dim=-1
                    )
                    best_logp = (
                        logp_targets[0]
                        if positions[0] != blank
                        else logp_targets[1]
                    )

                    # Extend hyp by  selection
                    for j in range(logp_targets.size(0)):

                        # hyp
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"]
                            + logp_targets[j],
                            "hidden_dec": a_best_hyp["hidden_dec"],
                        }

                        if positions[j] == self.blank_id:
                            beam_hyps.append(topk_hyp)
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = a_best_hyp["hidden_lm"]
                            continue

                        if logp_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(positions[j].item())
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = hidden_lm
                                topk_hyp["logp_score"] += (
                                    self.lm_weight
                                    * log_probs_lm[0, 0, positions[j]]
                                )
                            process_hyps.append(topk_hyp)
            # Add norm score
            nbest_hyps = sorted(
                beam_hyps,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[: self.nbest]
            all_predictions = []
            all_scores = []
            for hyp in nbest_hyps:
                all_predictions.append(hyp["prediction"][1:])
                all_scores.append(hyp["logp_score"] / len(hyp["prediction"]))
            nbest_batch.append(all_predictions)
            nbest_batch_score.append(all_scores)
        return (
            [nbest_utt[0] for nbest_utt in nbest_batch],
            torch.Tensor(
                [nbest_utt_score[0] for nbest_utt_score in nbest_batch_score]
            )
            .exp()
            .mean(),
            nbest_batch,
            nbest_batch_score,
        )

    def _lm_forward_step(self, inp_tokens, memory):
        """This method should implement one step of
        forwarding operation for language model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
            (e.g., RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        hs : No limit
            The memory variables are generated in this timestep.
            (e.g., RNN hidden states).
        """
        with torch.no_grad():
            logits, hs = self.lm(inp_tokens, hx=memory)
            log_probs = self.softmax(logits)
        return log_probs, hs

    def _get_sentence_to_update(self, selected_sentences, input_PN):
        """Select and return the updated hiddens and output
        from the Prediction Network.

        Arguments
        ----------
        selected_sentences : list
            List of updated sentences (indexes).
        output_PN: torch.tensor
            Output tensor from prediction network (PN).

        Returns
        -------
        selected_input_PN: torch.tensor
            Outputs a logits tensor [B_selected, U].
        """
        selected_input_PN = []
        for i in selected_sentences:
            selected_input_PN.append(input_PN[i].view(-1))
        selected_input_PN = pad_sequence(selected_input_PN, batch_first=True, padding_value=0.)

        return selected_input_PN

    def _forward_PN(self, out_PN, out_TN):
        """Compute forward-pass through a list of prediction network (PN) layers.

        Arguments
        ----------
        out_PN : torch.tensor
            Input sequence from prediction network with shape
            [batch, target_seq_lens].
        out_TN : torch.tensor
            Output sequence from transcription network with shape
            [batch, chuncks, source_seq_lens, dims].

        Returns
        -------
        out_PN : torch.tensor
            Outputs a logits tensor [B,U, hiddens].
        """
        out_PN, _ = self.model.decode(out_PN, out_TN)

        return out_PN
