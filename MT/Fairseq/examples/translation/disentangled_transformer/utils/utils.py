import os
import sys
from pathlib import Path
import importlib
import math
from typing import Tuple
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fairseq import utils
from collections import Counter
import nltk
import string

from sklearn.manifold import Isomap, TSNE
import pandas as pd

# Solve relative import
# https://gist.github.com/vaultah/d63cb4c86be2774377aa674b009f759a
file = Path(__file__).resolve()
parent, top = file.parent, file.parents[2]

sys.path.append(str(top))
__package__ = ".".join(parent.parts[len(top.parts):])
importlib.import_module(__package__)

from .. import DisentangledTransformerModel
from fairseq.disentangled_transformer.modules.disentangled_attention import DisentangledAttention

nltk.download("stopwords")

def load_model(ckpt_file, data_path, bpe_code, bpe="subword_nmt", grad_multiplier=10.):
    ckpt_dir = os.path.dirname(ckpt_file)
    ckpt_file = os.path.basename(ckpt_file)

    model = DisentangledTransformerModel.from_pretrained(
        ckpt_dir,
        checkpoint_file=ckpt_file,
        data_name_or_path=data_path,
        bpe=bpe,
        bpe_codes=bpe_code,
        grad_multiplier=grad_multiplier
    )

    return model

def model_trainable_parameters(model):
    params = 0
    for param in model.parameters():
        if param.requires_grad:
            params += param.numel()

    return params

def dataset_iterator(model, split="valid", epoch=1):
    if split == "train":
        split = model.cfg.dataset.train_subset
    elif split == "valid":
        split = model.cfg.dataset.valid_subset
    else:
        split = model.cfg.dataset.gen_subset

    model.task.load_dataset(
        split,
        epoch=epoch,
        combine=False,
        data_selector=None,
        tpu=True,
    )

    iterator = model.task.get_batch_iterator(
        dataset=model.task.dataset(split),
        max_tokens=model.cfg.dataset.max_tokens,
        max_sentences=model.cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            model.task.max_positions(),
            model.max_positions,
            model.cfg.dataset.max_tokens,
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=model.cfg.dataset.required_batch_size_multiple,
        seed=model.cfg.common.seed,
        num_shards=1,
        shard_id=0,
        num_workers=model.cfg.dataset.num_workers,
        epoch=1,
        data_buffer_size=model.cfg.dataset.data_buffer_size
    ).next_epoch_itr(
        fix_batches_to_gpus=model.cfg.distributed_training.fix_batches_to_gpus,
        shuffle=False
    )

    return iterator

def get_batch_data_with_length_limited(model, iterator, min_length=1):
    data = None

    for batch in iterator:
        _, _, _, src_lens, tgt_lens = decode_model_inputs(model, batch)

        if min(src_lens) > min_length and min(tgt_lens) > min_length:
            data = batch
            break

    return data

def decode_model_inputs(model, data):
    src_tokens = []
    prev_tokens = []
    tgt_tokens = []

    src_lens = []
    tgt_lens = []

    for batch in data["net_input"]["src_tokens"]:
        src_tokens.append([model.src_dict.string([token]) for token in list(batch)])
        src_lens.append(src_tokens[-1].index(""))
        src_lens.append(src_tokens[-1].index(""))
        src_tokens[-1] = src_tokens[-1][:src_lens[-1]]

    for batch_tgt, batch_prev in zip(data["target"], data["net_input"]["prev_output_tokens"]):
        tgt_tokens.append([model.tgt_dict.string([token]) for token in list(batch_tgt)])
        tgt_tokens[-1][tgt_tokens[-1].index("")] = "<eos>"
        tgt_lens.append(tgt_tokens[-1].index("<eos>") + 1)
        tgt_tokens[-1] = tgt_tokens[-1][:tgt_lens[-1]]

        prev_tokens.append([model.tgt_dict.string([token]) for token in list(batch_prev)])
        prev_tokens[-1] = ["<sos>"] + prev_tokens[-1][1:]
        prev_tokens[-1] = prev_tokens[-1][:tgt_lens[-1]]

    return src_tokens, prev_tokens, tgt_tokens, src_lens, tgt_lens

def model_inference(model, data):
    with torch.no_grad():
        model.models[0].eval()
        outputs = model.models[0](**{k:v.to(model.device) for k, v in data["net_input"].items()}, debug=True)

    return outputs

def plot_attention_map(attn: torch.Tensor, tgt_label="auto", src_label="auto", save=""):
    """
        attn: [heads, layers, tgt, src]
    """
    attn = attn.detach().cpu()
    layers = attn.shape[1]
    
    plt.figure(figsize=(9, 12))
    for layer in range(layers):
        plt.subplot(math.ceil(layers/2), 2, layer+1)
        sns.heatmap(attn[:, layer].mean(0), xticklabels=src_label, yticklabels=tgt_label)
        plt.title("Layer {}".format(layer+1))

    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0)

    plt.show()

def plot_cluster_assignment(outputs_dict, src_tokens, tgt_tokens, enc=False, dec=False, dec_enc=False,
                            query=False, key=False, mi=False, batch=0, head=0, figure_dir=""):
    assert enc or dec or dec_enc
    assert query or key
    
    key = ""
    if enc:
        key += "enc_"
    elif dec:
        key += "dec_"
    else:
        key += "dec_enc_"

    if mi:
        key += "mi_cluster_probs"
        if enc:
            tokens = src_tokens[batch]
        else:
            tokens = tgt_tokens[batch]
    else:
        if query:
            key += "cluster_probs_q"
            if enc:
                tokens = src_tokens[batch]
            else:
                tokens = tgt_tokens[batch]
        else:
            key += "cluster_probs_k"
            if dec:
                tokens = tgt_tokens[batch]
            else:
                tokens = src_tokens[batch]

    cluster_prob = outputs_dict[key][batch, head, :, :len(tokens)]
    pred_cluster = cluster_prob.max(dim=-1).indices
    layers = cluster_prob.shape[0]

    plt.figure(figsize=(16, 8))
    for layer in range(layers):                
        plt.subplot(math.ceil(layers/2), 2, layer+1)
        plt.title("Layer {}. Semantic cluster assignment probs.".format(layer+1))
        sns.heatmap(cluster_prob[layer].T, xticklabels=tokens, yticklabels="auto")
        plt.ylabel("Cluster id")

        print("Layer ", layer)
        for c in range(4):
            print("Cluster", c, np.array(tokens)[pred_cluster[layer] == c])

    plt.tight_layout()

    if figure_dir:
        if mi:
            file = os.path.join(figure_dir, "mi_cluster_probs_b_{}_h_{}.png".format(batch, head))
        else:
            file = os.path.join(figure_dir, "semantic_cluster_probs_b_{}_h_{}.png".format(batch, head))
        plt.savefig(file, bbox_inches="tight", pad_inches=0)
        print("Save figure to ", file)

def plot_token_distribution(embeddings: torch.Tensor, head_idx: torch.Tensor,
                            save: str=""):
    """
        embeddings: [tokens, dims]
        head_idx: [clusters,]
        save: Path to save figure
    """
    sns.set_style("white")

    head_idx = head_idx.int().numpy().reshape(-1, 1)
    clusters = head_idx.max() + 1

    tsne = TSNE(n_components=2)    
    tsne_results = tsne.fit_transform(embeddings)

    df_result = pd.DataFrame(
        np.concatenate([tsne_results, head_idx], 1),
        columns=(["x", "y", "Head index"])
    )

    f, ax = plt.subplots(figsize=(9, 6))
    palette = np.array(sns.color_palette("Set2", clusters))
    palette = list(palette[np.unique(df_result.iloc[:, 2].astype(np.int32))])

    ax = sns.scatterplot(
        x='x', # Horizontal axis
        y='y', # Vertical axis
        hue="Head index",
        style="Head index",
        data=df_result.iloc[:, :3].astype(np.float32), # Data source
        sizes=[0.1]*(max(head_idx)+1),
        palette=palette,
        ax=ax
    ) # size and dimension
    plt.legend(loc="upper right", title="Head index")

    # plt.title("Query distribution")

    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight", pad_inches=0.1)
        print("Save figure to ", save)

def plot_semantic_mask(outputs_dict, src_tokens, tgt_tokens, batch=0, head=0,
                       enc=False, dec=False, dec_enc=False, figure_dir=""):
    assert enc or dec or dec_enc

    if enc:
        semantic_mask = outputs_dict["enc_mask"][batch, head]
        tgt_tokens = src_tokens
    elif dec:
        semantic_mask = outputs_dict["dec_mask"][batch, head]
        src_tokens = tgt_tokens
    else:
        semantic_mask = outputs_dict["dec_enc_mask"][batch, head]

    layers = semantic_mask.shape[0]

    plt.figure(figsize=(12, 16))
    for layer in range(layers):                
        plt.subplot(math.ceil(layers/2), 2, layer+1)
        plt.title("Layer {}. Semantic mask.".format(layer+1))
        sns.heatmap(semantic_mask[layer], xticklabels=src_tokens[batch], yticklabels=tgt_tokens[batch])

    plt.tight_layout()

    if figure_dir:
        file = os.path.join(figure_dir, "semantic_mask_b_{}_h_{}.png".format(batch, head))
        plt.savefig(file, bbox_inches="tight", pad_inches=0)
        print("Save figure to ", file)

def plot_cluster_token_distribution(model, iterator, layer=-1, head=0, enc=False, dec=False, dec_enc=False,
                                    query=False, key=False, figure_dir=""):
    assert enc or dec or dec_enc
    assert query or key

    stopwords = list(nltk.corpus.stopwords.words("english"))

    if enc:
        key = "enc_"
    elif dec:
        key = "dec_"
    else:
        key = "dec_enc_"

    if query:
        key += "cluster_probs_q"
    else:
        key += "cluster_probs_k"

    cluster_token_dist = {}

    for idx, batch in enumerate(iterator):
        print("\rCollecting examples ({}/{}) ...".format(idx+1, len(iterator)), end="")

        src_tokens, prev_tokens, tgt_tokens, src_lens, tgt_lens = decode_model_inputs(model, batch)

        if enc:
            tokens = src_tokens
        elif dec_enc:
            if query:
                tokens = tgt_tokens
            else:
                tokens = src_tokens
        else:
            tokens = tgt_tokens

        for m in model.models[0].modules():
            if isinstance(m, DisentangledAttention):
                m.outputs_dict = {}

        outputs = model_inference(model, batch)
        cluster_indices = outputs[1][key][:, head, layer].max(dim=-1).indices

        for b in range(len(cluster_indices)):
            for l in range(len(tokens[b])):
                counter = cluster_token_dist.get(
                    int(cluster_indices[b][l]), Counter()
                )
                
                tokens[b][l] = tokens[b][l].replace("&quot;", '"').replace("&apos;", "'") 

                if not tokens[b][l].lower() in stopwords and not tokens[b][l] in string.punctuation:
                    if not "<pad>" in tokens[b][l] and not "<eos>" in tokens[b][l] and not "<sos>" in tokens[b][l]:
                        counter.update([tokens[b][l]])
                        cluster_token_dist[int(cluster_indices[b][l])] = counter

    print("Plotting token dstribution ...")
    for c in range(len(cluster_token_dist)):
        counter = cluster_token_dist[c]
        results = pd.DataFrame(
            [[token, int(count)] for (token, count) in counter.most_common(15)],
            columns=["Token", "Count"],
            dtype=None
        )
    
        plt.figure(figsize=(20, 3))
        sns.barplot(data=results, x="Token", y="Count", ci=0, palette="rocket_r")
        # plt.title("Cluster {}. Token distribution.".format(c+1))

        if figure_dir:
            file = os.path.join(figure_dir, "token_dist_h_{}_c_{}_l_{}.png".format(head, c, layer))
            plt.savefig(file, bbox_inches="tight", pad_inches=0.1)
            print("Save figure to ", file)

def js_div(p: torch.Tensor, q: torch.Tensor):
    """
    Parameters:
        p: [d1, ..., dn, dims]
        q: [d1, ..., dn, dims]

    Returns:
        [d1, ..., dn]
    """
    m = 0.5 * (p + q)
    js = 0.5 * p * (torch.log2(p + 1e-6) - torch.log2(m + 1e-6)) + 0.5 * q * (torch.log2(q + 1e-6) - torch.log2(m + 1e-6))
    js = js.sum(-1)

    return js

def entropy(p: torch.Tensor):
    """
    Parameters:
        p: [d1, ..., dn, dims]

    Returns:
        [d1, ..., dn]
    """
    return -(p * torch.log2(p + 1e-6)).sum(-1)

def attention_redundancy(attention: torch.Tensor, head=False, layer=False)-> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters:
        attention: [batch, layers, heads, tgt length, src length]

    Returns:
        (redundancy score, redundancy matrix) -- 
            redundancy score: [1,]
            redundancy matrix: 
                head = False and layer = False:
                    [batch, layers, heads]
                layer = True:
                    [batch, layers]
                head = True:
                    [batch, layers*heads, layers*heads]
    """    
    B, L, H, L_TGT, L_SRC = attention.size()

    assert head or layer

    if layer:
        # redundancy of each layer
        redundancy = (math.log2(H) - (entropy(attention.mean(2)) - entropy(attention).mean(2)).mean(2))/math.log2(H)
        score = redundancy.mean((1))
    else:
        # redundancy between all heads
        redundancy = 1 - js_div(attention.view(B, L*H, 1, L_TGT, L_SRC), attention.view(B, 1, L*H, L_TGT, L_SRC)).mean(-1)
        score = redundancy.mean((1, 2))

    return score, redundancy

@torch.no_grad()
def redundancy_evaluate(model, iterator):
    layer_rdc = []
    head_rdc = []

    for m in model.models[0].modules():
        if isinstance(m, DisentangledAttention):
            m.debug = True

    for idx, batch in enumerate(iterator):
        print("\rEvaluate batch {}".format(idx+1), end="")
        outputs = model.models[0](**{k:v.to(model.device) for k, v in batch["net_input"].items()}, debug=True)

        enc_attn = outputs[1]["enc_mask_attn"].permute(0, 2, 1, 3, 4).contiguous()
        dec_attn = outputs[1]["dec_mask_attn"].permute(0, 2, 1, 3, 4).contiguous()
        dec_enc_attn = outputs[1]["dec_enc_mask_attn"].permute(0, 2, 1, 3, 4).contiguous()
        src_tokens, prev_tokens, tgt_tokens, src_lens, tgt_lens = decode_model_inputs(model, batch)

        for attn, (src_len, tgt_len) in zip([enc_attn, dec_attn, dec_enc_attn], [(src_lens, src_lens), (tgt_lens, tgt_lens), (src_lens, tgt_lens)]):
            for b in range(len(attn)):
                _attn = enc_attn[b, :, :, :tgt_len[b], :src_len[b]]

                l_score, l_r = attention_redundancy(_attn[None], layer=True) # L
                h_score, h_r = attention_redundancy(_attn[None], head=True) # (LxH) x (LxH)

                layer_rdc.append(l_score.cpu().numpy())
                head_rdc.append(h_score.cpu().numpy())

        for m in model.models[0].modules():
            if isinstance(m, DisentangledAttention):
                m.outputs_dict = {}

    return np.mean(layer_rdc), np.mean(head_rdc)
