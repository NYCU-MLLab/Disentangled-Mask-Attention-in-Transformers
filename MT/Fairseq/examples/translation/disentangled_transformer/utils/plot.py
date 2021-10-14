import os
from argparse import ArgumentParser
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from utils import (
    load_model,
    model_trainable_parameters,
    dataset_iterator,
    get_batch_data_with_length_limited,
    decode_model_inputs,
    attention_redundancy,
    plot_token_distribution,
    plot_semantic_mask,
    plot_cluster_assignment,
    plot_cluster_token_distribution
)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "task", type=str,
        help="Support tasks: iwslt14_de_en, wmt17_zh_en, wmt14_en_de"
    )
    parser.add_argument(
        "model", type=str,
        help="Support models: disentangled_Transformer, transformer"
    )

    parser.add_argument(
        "--ckpt_dir", default="", type=str,
        help="Root folder of checkpoints including `model' folder."
    )
    parser.add_argument(
        "--ckpt", default="", type=str,
        help="Path of ckpt file"
    )
    parser.add_argument(
        "--code", default="", type=str,
        help="Path of BPE code"
    )
    parser.add_argument(
        "--data", default="", type=str,
        help="Path of biniarized data"
    )
    parser.add_argument(
        "-b", "--batch", default=0, type=int,
        help="Batch index"
    )
    parser.add_argument(
        "--head", default=0, type=str,
        help="Head index"
    )
    parser.add_argument(
        "-l", "--layer", default=5, type=int,
        help="Layer index"
    )

    parser.add_argument("--query_tsne", action="store_true")
    parser.add_argument("--head_redundancy", action="store_true")
    parser.add_argument("--semantic_clustering", action="store_true")
    parser.add_argument("--cluster_token_dist", action="store_true")
    parser.add_argument(
        "--enc", action="store_true",
        help="Plot cluster proability and cluster-token distribution of encoder attention."
    )
    parser.add_argument(
        "--dec", action="store_true",
        help="Plot cluster proability and cluster-token distribution of decoder attention."
    )
    parser.add_argument(
        "--dec_enc", action="store_true",
        help="Plot cluster proability and cluster-token distribution of decoder-encoder attention."
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()

def parse_paths(args):
    task = args.task
    if args.task == "iwslt14_de_en":
        task = "iwslt14.tokenized.de-en"

    if args.ckpt_dir:
        ckpt = os.path.join(args.ckpt_dir, f"{args.model}/ckpt/checkpoint_avg_last_5.pt")
    elif not args.ckpt:
        ckpt = f"checkpoints/{task}/{args.model}/ckpt/checkpoint_avg_last_5.pt"
    else:
        ckpt = args.ckpt

    if not args.code:
        code = f"examples/translation/{task}/code"
    else:
        code = args.code

    if not args.data:
        data = f"data-bin/{task}"
    else:
        data = args.data

    for path in [ckpt, code, data]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

    figure = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "figure")
    if not os.path.exists(figure):
        os.makedirs(figure)

    return ckpt, code, data, figure

def plot_head_redundancy(outputs_dict, src_lens, tgt_lens, enc=False, dec=False, dec_enc=False,
                         batch=0, figure_dir=""):
    assert enc or dec or dec_enc

    if enc:
        attn = outputs_dict["enc_mask_attn"][:, :, :, :src_lens[batch], :src_lens[batch]]
    elif dec:
        attn = outputs_dict["dec_mask_attn"][:, :, :, :tgt_lens[batch], :tgt_lens[batch]]
    else:
        attn = outputs_dict["dec_enc_mask_attn"][:, :, :, :tgt_lens[batch], :src_lens[batch]]
    attn = attn[batch:batch+1].transpose(2, 1).contiguous()

    h_score, h_r = attention_redundancy(attn, head=True) # (LxH) x (LxH)

    print("Head redundancy: {:.2f}".format(h_score[0]))
    plt.figure(figsize=(12, 4))
    sns.heatmap(h_r[0], vmin=0, vmax=1)
    plt.xlabel("Head index")
    plt.ylabel("Head index")

    if figure_dir:
        plt.savefig(os.path.join(figure_dir, "head_redundancy.png"), dpi=300, bbox_inches="tight", pad_inches=0)
        print("Save head redundancy to ", os.path.join(figure_dir, "head_redundancy.png"))

def plot_query_tsne(outputs_dict, enc=False, dec=False, dec_enc=False, layer=-1,
                    figure_dir=""):
    assert enc or dec or dec_enc

    if enc:
        embeddings = outputs_dict["enc_query"][:, :, layer]
    elif dec:
        embeddings = outputs_dict["dec_query"][:, :, layer]
    else:
        embeddings = outputs_dict["dec_enc_query"][:, :, layer]

    head_idx = torch.arange(embeddings.size()[1]).view(1, embeddings.size()[1], 1).repeat(
        embeddings.size()[0], 1, embeddings.size()[2]
    ).view(-1)

    print("\tSamples: ", len(embeddings))
    embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    print("\tTokens: ", len(embeddings))

    plot_token_distribution(
        embeddings,
        head_idx,
        save=os.path.join(figure_dir, "query_tsne.png") if figure_dir else ""
    )

def plot_semantic_clustering(outputs_dict, src_tokens, tgt_tokens, enc=False, dec=False, dec_enc=False,
                             batch=0, head=0, figure_dir=""):
    assert enc or dec or dec_enc

    plot_semantic_mask(
        outputs_dict, src_tokens, tgt_tokens, batch, head, enc, dec, dec_enc, figure_dir
    )

    plot_cluster_assignment(
        outputs_dict, src_tokens, tgt_tokens, enc=False, dec=True, dec_enc=False, query=True,
        batch=batch, head=head, figure_dir=figure_dir
    )

@torch.no_grad()
def main():
    args = parse_args()
    ckpt, code, data, figure = parse_paths(args)

    # Load model
    print("Loading model ...")
    model = load_model(ckpt, data, code)
    if not args.cpu:
        model = model.cuda()
    print("Parameters: ", format(model_trainable_parameters(model), ","), end="\n")

    # Load data iterator
    print("Loading dataset ...", end="\n")
    iterator = dataset_iterator(model.cpu(), split="test")
    batch_data = get_batch_data_with_length_limited(model, iterator, min_length=20)
    src_tokens, prev_tokens, tgt_tokens, src_lens, tgt_lens = decode_model_inputs(model, batch_data)

    print("Inferencing model ...")
    model.eval()
    outputs = model.models[0](**{k:v.to(model.device) for k, v in batch_data["net_input"].items()}, debug=True)
    outputs_dict = outputs[1]

    if args.head_redundancy or args.all:
        print("\nPlotting head redundancy ...")
        plot_head_redundancy(outputs_dict, src_lens, tgt_lens, enc=True, figure_dir=figure)

    if args.query_tsne or args.all:
        print("\nPlotting query distribution tSNE ...")
        plot_query_tsne(outputs_dict, dec=True, figure_dir=figure)

    if args.semantic_clustering or args.all:
        print("\nPlotting semantic clustering ...")
        plot_semantic_mask(
            outputs_dict, src_tokens, tgt_tokens,
            args.batch, args.head, enc=True, dec=False, dec_enc=False, figure_dir=figure
        )

        print("\nPlotting semantic cluster probablity distribution ...")
        plot_cluster_assignment(
            outputs_dict, src_tokens, tgt_tokens, enc=args.enc, dec=args.dec, dec_enc=args.dec_enc,
            query=True, batch=args.batch, head=args.head, figure_dir=figure
        )
        print("\nPlotting MI cluster probablity distribution ...")
        plot_cluster_assignment(
            outputs_dict, src_tokens, tgt_tokens, enc=args.enc, dec=args.dec, dec_enc=args.dec_enc,
            mi=True, query=True, batch=args.batch, head=args.head, figure_dir=figure
        )

    if args.cluster_token_dist or args.all:
        print("\nPlotting cluster-token distribution ...")
        plot_cluster_token_distribution(
            model, dataset_iterator(model, split="test"),
            layer=args.layer,
            head=args.head,
            enc=args.enc,
            dec=args.dec,
            dec_enc=args.dec_enc,
            query=True,
            key=False,
            figure_dir=figure
        )

if __name__ == "__main__":
    main()
