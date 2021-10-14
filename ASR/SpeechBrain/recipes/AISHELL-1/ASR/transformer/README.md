# AISHELL-1 ASR with Transformers.
This folder contains recipes for tokenization and speech recognition with [AISHELL-1](https://www.openslr.org/33/), a 150-hour Chinese ASR dataset.

### How to run
1- Train a tokenizer. The tokenizer takes in input the training transcripts and determines the subword units that will be used for both acoustic and language model training.

```
cd ../../Tokenizer
python train.py hparams/train_transformer_tokenizer_bpe5000.yaml --data_folder=/localscratch/aishell/
```
If not present in the specified data_folder, the dataset will be automatically downloaded there.
This step is not mandatory. We will use the official tokenizer downloaded from the web if you do not 
specify a different tokenizer in the speech recognition recipe. 

2- Train the speech recognizer
```
python train.py hparams/train_ASR_transformer.yaml --data_folder=/localscratch/aishell/
```

# PreTrained Model + Easy-Inference
You can find the pre-trained model with an easy-inference function on HuggingFace
- https://huggingface.co/speechbrain/asr-transformer-aishell
- https://huggingface.co/speechbrain/asr-wav2vec2-transformer-aishell
 

# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```