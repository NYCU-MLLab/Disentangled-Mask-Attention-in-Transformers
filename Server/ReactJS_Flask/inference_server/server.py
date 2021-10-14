import argparse
import os
import io
import json
import base64
import torch
import torchaudio
from argparse import ArgumentParser
from typing import Dict, List
from flask import Flask, jsonify, Response, request
from flask_cors import CORS
import matplotlib.pyplot as plt
import seaborn as sns
import jieba
from opencc import OpenCC
from matplotlib.font_manager import FontProperties
from pydub import AudioSegment

# Fairseq imports
from fairseq.disentangled_transformer.modules.disentangled_attention import DisentangledAttention
from fairseq.disentangled_transformer.utils.utils import (
    load_model,
)
from fairseq.hub_utils import GeneratorHubInterface

# SpeechBrain imports
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from asr import ASR
from speechbrain.nnet.attention import DisentangledMaskAttention

app = Flask(__name__)
CORS(app)

mt_models: Dict[str, GeneratorHubInterface] = {}
asr_model = None
cc = OpenCC("t2s")
font = FontProperties(fname="./NotoSansCJK-Regular.ttc")

def array_to_img_base64(array, src_tokens=[], tgt_tokens=[]):
    sns.heatmap(array)

    if src_tokens:
        plt.xticks(range(array.shape[1]), src_tokens, fontproperties=font, rotation=90)

    if tgt_tokens:
        plt.yticks(range(array.shape[0]), tgt_tokens, fontproperties=font, rotation=0)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf8")

def get_asr_model_imgs(model, head, layer, src_tokens=[], tgt_tokens=[]):
    outputs = {}

    for idx, encoder in enumerate(model.hparams.Transformer.encoder.layers):
        if idx == layer:
            outputs["enc_attn"] = array_to_img_base64(
                encoder.self_att.outputs_dict["attn"][0, head],
                src_tokens, src_tokens
            )
            outputs["enc_mask"] = array_to_img_base64(
                encoder.self_att.outputs_dict["mask"][0, head],
                src_tokens, src_tokens
            )
            outputs["enc_mask_attn"] = array_to_img_base64(
                encoder.self_att.outputs_dict["mask_attn"][0, head],
                src_tokens, src_tokens
            )
        
    for idx, decoder in enumerate(model.hparams.Transformer.decoder.layers):
        if idx == layer:
            outputs["dec_attn"] = array_to_img_base64(
                decoder.self_attn.outputs_dict["attn"][0, head],
                tgt_tokens, tgt_tokens
            )
            outputs["dec_mask"] = array_to_img_base64(
                decoder.self_attn.outputs_dict["mask"][0, head],
                tgt_tokens, tgt_tokens
            )
            outputs["dec_mask_attn"] = array_to_img_base64(
                decoder.self_attn.outputs_dict["mask_attn"][0, head],
                tgt_tokens, tgt_tokens
            )

            outputs["dec_enc_attn"] = array_to_img_base64(
                decoder.mutihead_attn.outputs_dict["attn"][0, head],
                src_tokens, tgt_tokens
            )
            outputs["dec_enc_mask"] = array_to_img_base64(
                decoder.mutihead_attn.outputs_dict["mask"][0, head],
                src_tokens, tgt_tokens
            )
            outputs["dec_enc_mask_attn"] = array_to_img_base64(
                decoder.mutihead_attn.outputs_dict["mask_attn"][0, head],
                src_tokens, tgt_tokens
            )

    return outputs

def get_mt_model_imgs(model, head, layer, src_tokens=[], tgt_tokens=[]):
    outputs = {}

    for idx, encoder in enumerate(model.models[0].encoder.layers):
        if idx == layer:
            outputs["enc_attn"] = array_to_img_base64(
                encoder.self_attn.attn.outputs_dict["attn"][0, head],
                src_tokens, src_tokens
            )
            outputs["enc_mask"] = array_to_img_base64(
                encoder.self_attn.attn.outputs_dict["mask"][0, head],
                src_tokens, src_tokens
            )
            outputs["enc_mask_attn"] = array_to_img_base64(
                encoder.self_attn.attn.outputs_dict["mask_attn"][0, head],
                src_tokens, src_tokens
            )

    for idx, decoder in enumerate(model.models[0].decoder.layers):
        if idx == layer:
            outputs["dec_attn"] = array_to_img_base64(
                decoder.self_attn.attn.outputs_dict["attn"][0, head],
                tgt_tokens, tgt_tokens
            )
            outputs["dec_mask"] = array_to_img_base64(
                decoder.self_attn.attn.outputs_dict["mask"][0, head],
                tgt_tokens, tgt_tokens
            )
            outputs["dec_mask_attn"] = array_to_img_base64(
                decoder.self_attn.attn.outputs_dict["mask_attn"][0, head],
                tgt_tokens, tgt_tokens
            )

            outputs["dec_enc_attn"] = array_to_img_base64(
                decoder.encoder_attn.attn.outputs_dict["attn"][0, head],
                src_tokens, tgt_tokens
            )
            outputs["dec_enc_mask"] = array_to_img_base64(
                decoder.encoder_attn.attn.outputs_dict["mask"][0, head],
                src_tokens, tgt_tokens
            )
            outputs["dec_enc_mask_attn"] = array_to_img_base64(
                decoder.encoder_attn.attn.outputs_dict["mask_attn"][0, head],
                src_tokens, tgt_tokens
            )

    return outputs

@app.route("/inference/mt/<lang>/", methods=["POST", "GET"])
def mt_inference(lang):
    if request.method == "POST":
        data = request.get_json()

        if lang == "ZhEn":
            data["sentence"] = " ".join(jieba.cut(cc.convert(data["sentence"])))

        tokenized_sentences = [mt_models[lang].encode(data["sentence"])]
        batched_hypos = mt_models[lang].generate(
            tokenized_sentences,
            beam=1,
            verbose=False,
            debug=True
        )
        result = mt_models[lang].string(batched_hypos[0][0]["tokens"])
        result = mt_models[lang].remove_bpe(result)
        result = mt_models[lang].detokenize(result)

        src_tokens = [mt_models[lang].src_dict[int(token)] for token in tokenized_sentences[0]]
        tgt_tokens = [mt_models[lang].tgt_dict[int(token)] for token in batched_hypos[0][0]["tokens"]]

        result = {"result": result}
        result.update(get_mt_model_imgs(
            mt_models[lang], int(data["head"]), int(data["layer"]),
            src_tokens, tgt_tokens
        ))

        return jsonify(result)
    else:
        sentence = request.args.get("sentence")

        if lang == "ZhEn":
            sentence = " ".join(jieba.cut(cc.convert(sentence)))

        result = mt_models[lang].translate([sentence])[0]
        return "{}".format(result)

@app.route("/inference/asr/", methods=["POST"])
def asr_inference():
    if request.method == "POST":
        data = request.get_json()

        audio = data["audio"]
        fileformat = "wav"
        filename = "_audio.{}".format(fileformat)
        AudioSegment.from_file(io.BytesIO(base64.b64decode(audio.split(",")[1]))).export(
            filename, format=fileformat
        )
        waveform, _ = torchaudio.load(filename)

        result_tokens, result = asr_model.compute_forward(waveform.view((1, -1)), torch.as_tensor([1.]))
        
        results = {"result": result}
        results.update(get_asr_model_imgs(
            asr_model, int(data["head"]), int(data["layer"]), src_tokens=[], tgt_tokens=result_tokens
        ))
    
        return jsonify(results)

def init_mt_model():
    global mt_models

    configs = json.load(open("configs.json", "r"))

    for lang in configs["mt"].keys():
        print("Loading MT ({}) model ...".format(lang))
        print(configs["mt"][lang]["data_path"])
        mt_model = load_model(
            os.path.abspath(configs["mt"][lang]["ckpt_file"]),
            os.path.abspath(configs["mt"][lang]["data_path"]),
            os.path.abspath(configs["mt"][lang]["bpe_code"])
        )
        for m in mt_model.models[0].modules():
            if isinstance(m, DisentangledAttention):
                m.debug = True
        mt_models[lang] = mt_model
        print("Finished")

def init_asr_model():
    global asr_model

    configs = json.load(open("configs.json", "r"))

    hparams_file, run_opts, overrides = sb.parse_arguments(
        [configs["asr"]["aishell"]["config"],
         "--data_folder=.",
         "--output_folder="+os.path.dirname(
             os.path.dirname(configs["asr"]["aishell"]["ckpt_dir"])
         ),
         "--test_beam_size=1"]
    )
    run_opts["device"]="cpu"

    print("Loading ASR model ...") 
    with open(hparams_file, "r") as file:
        hparams = load_hyperpyyaml(file, overrides)

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    asr_model = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"]
    )

    tokenizer = hparams["tokenizer"]
    asr_model.tokenizer = tokenizer

    ckpt = asr_model.checkpointer.find_checkpoint(
        ckpt_predicate=lambda x: os.path.normpath(x.path) == os.path.normpath(configs["asr"]["aishell"]["ckpt_dir"])
    )
    asr_model.checkpointer.load_checkpoint(ckpt, device=run_opts["device"])
    asr_model.hparams.model.eval()

    for m in asr_model.hparams.Transformer.modules():
        if isinstance(m, DisentangledMaskAttention):
            m.debug = True
    print("Finished")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--ckpt-file",
        default="/home/code/handoff/Disentangled-Transformer-MT-Clean/checkpoints/iwslt14.tokenized.de-en-35_11/disentangled_transformer/ckpt/checkpoint_avg_last_5.pt"
    )
    parser.add_argument(
        "--data-path",
        default="/home/code/handoff/Disentangled-Transformer-MT-Clean/data-bin/iwslt14.tokenized.de-en"
    )
    parser.add_argument(
        "--code",
        default="/home/code/handoff/Disentangled-Transformer-MT-Clean/examples/translation/iwslt14.tokenized.de-en/code"
    )

    return parser.parse_args()

def main():
    args = parse_args()
    init_asr_model()
    init_mt_model()

    app.run(port=8888, host="0.0.0.0")

if __name__ == "__main__":
    main()
