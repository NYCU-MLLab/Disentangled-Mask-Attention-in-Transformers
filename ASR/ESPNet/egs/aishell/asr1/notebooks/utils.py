
import math
from typing import Tuple
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE, Isomap
import pandas as pd

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

def plot_token_distribution(text_embedding, tokens, pred_cluster_idx,
                            auxiliary_text_embedding=None, auxiliary_tokens=None, auxiliary_cluster_idx=None,
                            cluster_centers=None, cluster_log_var=None, plot_token_label=False, cluster_label_str="Cluster id",
                            n_neighbors=5, log_scale=False, tsne=False, save="", marker=None):
    """
    Reference: https://stackoverflow.com/questions/46027653/adding-labels-in-x-y-scatter-plot-with-seaborn
        text_embeddings: [tokens, dims]
        cluster_idx: [clusters,]
        cluster_centers: [clusters, dims]
    """
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(float(point['x'])+0.05, float(point['y'])+0.05, str(point['val']))

    sns.set()
    cluster_idx = pred_cluster_idx.int().numpy().reshape(-1, 1)
    text_embedding_size = text_embedding.size()[0]
    markers = ["o"] * text_embedding_size
    sizes = [0.1] * text_embedding_size

    if auxiliary_text_embedding is not None:
        text_embedding = torch.cat([text_embedding, auxiliary_text_embedding])
        auxiliary_cluster_idx = auxiliary_cluster_idx.numpy().reshape(-1, 1)
        cluster_idx = np.concatenate([cluster_idx, auxiliary_cluster_idx])
        tokens = tokens + auxiliary_tokens
        markers += ["s"] * text_embedding_size
        sizes  += [0.1] * text_embedding_size
    else:
        auxiliary_text_embedding = []

    if cluster_centers is not None:
        clusters = cluster_centers.size()[0]
        normal = torch.distributions.normal.Normal(torch.from_numpy(cluster_centers.numpy()), cluster_log_var.exp().sqrt())
        cluster_centers_sampled = torch.cat([normal.sample().unsqueeze(0)]*100).view(100*clusters, -1)

        text_embedding = torch.cat([text_embedding, cluster_centers_sampled, cluster_centers])
        cluster_idx = np.concatenate([cluster_idx, torch.tensor([[i for i in range(clusters)]*100]).view(-1, 1), torch.tensor([[i] for i in range(clusters)])])
        tokens = tokens + ["cluster " + str(i) for i in range(clusters) for _ in range(100)] + ["cluster " + str(i) for i in range(clusters)]
        markers += ["h"] * clusters
        sizes  += [1] * clusters
    else:
        clusters = pred_cluster_idx.max() + 1

    if tsne:
        tsne = TSNE(n_components=2)    
        results = tsne.fit_transform(text_embedding)
    else:
        isomap = Isomap(n_components=2, n_neighbors=n_neighbors)    
        results = isomap.fit_transform(text_embedding)

    df_result = pd.DataFrame(
        np.concatenate([results, cluster_idx, np.array(tokens).reshape(-1, 1)], 1),
        columns=(["x", "y", cluster_label_str, "token"])
    )

    f, ax = plt.subplots(figsize=(9, 6))
    if cluster_centers is not None:
        start_idx = text_embedding_size + len(auxiliary_text_embedding)
        ax = sns.kdeplot(
            x="x",
            y="y",
            hue=cluster_label_str,
            data=df_result.iloc[start_idx:, :3].astype(np.float32),
            legend=False,
            levels=10,
            palette=sns.color_palette("Set2", clusters),
            alpha=0.25
        )
        # label_point(df_result.iloc[-clusters:].x, df_result.iloc[-clusters:].y, df_result.iloc[-clusters:].token, ax)
    
    ax = sns.scatterplot(
        x='x', # Horizontal axis
        y='y', # Vertical axis
        hue=cluster_label_str,
        data=df_result.iloc[:text_embedding_size, :3].astype(np.float32), # Data source
        style=cluster_label_str,
        palette=list(np.array(sns.color_palette("Set2", clusters))[np.unique(df_result.iloc[:text_embedding_size, 2].astype(np.int32))]),
        ax=ax
    ) # size and dimension

    if plot_token_label:
        label_point(df_result.iloc[:text_embedding_size].x, df_result.iloc[:text_embedding_size].y, df_result.iloc[:text_embedding_size].token, ax)

    if auxiliary_text_embedding is not None:
        end_idx = text_embedding_size + len(auxiliary_text_embedding)
        ax = sns.scatterplot(
            x='x', # Horizontal axis
            y='y', # Vertical axis
            hue=cluster_label_str,
            data=df_result.iloc[text_embedding_size:end_idx, :3].astype(np.float32), # Data source
            markes=marker,
            sizes=[0.1]*(max(cluster_idx)+1),
            ax=ax,
            legend=False,
            palette=list(np.array(sns.color_palette("Set2", clusters))[np.unique(df_result.iloc[text_embedding_size:end_idx, 2].astype(np.int32))])
        ) # size and dimension


        if plot_token_label:
            label_point(df_result.iloc[text_embedding_size:end_idx].x, df_result.iloc[text_embedding_size:end_idx].y, df_result.iloc[text_embedding_size:end_idx].token, ax)

    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")


    plt.title("Embedding distribution")

    if save:
        plt.savefig(save, dpi=300)

    plt.show() 

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

    if not head and not layer:
        # redundancy of each attention matrix
        redundancy = (attention.matmul(attention.transpose(4, 3)) - torch.eye(L_TGT).to(attention)).pow(2).sum(4).mean(3)
        score = redundancy.mean((1, 2))
    elif layer:
        # redundancy of each layer
        redundancy = (math.log2(H) - (entropy(attention.mean(2)) - entropy(attention).mean(2)).mean(2))/math.log2(H)
        score = redundancy.mean((1))
    else:
        # redundancy between all heads
        redundancy = 1 - js_div(attention.view(B, L*H, 1, L_TGT, L_SRC), attention.view(B, 1, L*H, L_TGT, L_SRC)).mean(-1)
        score = redundancy.mean((1, 2))

    return score, redundancy
    
def calculate_redundancy(model):
    encoder_attn = []
    decoder_attn = []
    dec_enc_attn = []

    for encoder in model.encoder.encoders:
        encoder_attn.append(encoder.self_attn.attn)
    for decoder in model.decoder.decoders:
        decoder_attn.append(decoder.self_attn.attn)
        dec_enc_attn.append(decoder.src_attn.attn)

    encoder_attn = torch.stack(encoder_attn, dim=1)
    decoder_attn = torch.stack(decoder_attn, dim=1)
    dec_enc_attn = torch.stack(dec_enc_attn, dim=1)
    attns = [encoder_attn, decoder_attn, dec_enc_attn]

    attn_rdc = []
    layer_rdc = []
    head_rdc = []
    for attn in attns:
        for b in range(len(attn)):
            _attn = attn[b]

            score, r = attention_redundancy(_attn[None]) # L x H
            l_score, l_r = attention_redundancy(_attn[None], layer=True) # L
            h_score, h_r = attention_redundancy(_attn[None], head=True) # (LxH) x (LxH)

            attn_rdc.append(score.cpu().numpy())
            layer_rdc.append(l_score.cpu().numpy())
            head_rdc.append(h_score.cpu().numpy())

    return np.mean(attn_rdc), np.mean(layer_rdc), np.mean(head_rdc)