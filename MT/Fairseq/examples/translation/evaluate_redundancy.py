#!/bin/python3

from disentangled_transformer.utils.utils import *
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_file", default="./checkpoints/iwslt14.tokenized.de-en/disentangled_transformer/ckpt/checkpoint_avg_last_5.pt")
    parser.add_argument("--data", default="./data-bin/iwslt14.tokenized.de-en")
    parser.add_argument("--code", default="./examples/translation/iwslt14.tokenized.de-en/code")
    args = parser.parse_args()

    ckpt_file = os.path.abspath(args.ckpt_file)
    data = os.path.abspath(args.data)
    code = os.path.abspath(args.code)

    model = load_model(ckpt_file, data, code).cuda()
    print("Trainable parameters: ", format(model_trainable_parameters(model), ","))

    iterator = dataset_iterator(model, split="test")
    lr, hr = redundancy_evaluate(model, iterator)

    print("\nLayer redundancy: ", format(lr, ".2f"))
    print("Head redundancy: ", format(hr, ".2f"))

    with open(os.path.join(os.path.dirname(ckpt_file), "redundancy.txt"), "w") as file:
        file.write("Layer redundancy: {:.2f}\n".format(lr, ".2f"))
        file.write("Head redundancy: {:.2f}\n".format(hr, ".2f"))

if __name__ == "__main__":
    main()