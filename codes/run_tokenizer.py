# tokenize the input text based on bert-base-chinese
import json
from pathlib import Path
from transformers import AutoTokenizer
import torch
from transformers import EncoderDecoderModel
from torch.utils.data import DataLoader
from transformers import AdamW
import numpy as np
from hanziconv import HanziConv
import argparse
import time
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments


### Retrieve load the dataset from the corresponding files
def get_dataset(questionfile):
    with open(questionfile, "r", encoding="utf-8") as fd:
        questions = [line.rstrip() for line in fd.readlines()]

    return questions


def show_answer(tokenizer_model, encodings, idx):
    print("Index :: ", idx)
    print("Text (Tokenized) :: ", tokenizer_model.decode(encodings['input_ids'][idx]))
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument("--textfile", type=str, required=True, help="File")
    parser.add_argument("--splitfile", type=str, required=True, help="File")

    args = parser.parse_args()
    print("Get all Arguments:")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    modelpath = "bert-base-chinese"
    # modelpath = "adamlin/bert-distil-chinese"

    texts = get_dataset(args.textfile)

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]"]})
    encodings = tokenizer(texts, truncation=True, padding=True)

    print("Show some examples: ")
    show_answer(tokenizer, encodings, 0)
    show_answer(tokenizer, encodings, 100)
    show_answer(tokenizer, encodings, 2000)
    show_answer(tokenizer, encodings, -1)
    print("Length of Train Set: {}".format(len(texts)))

    with open(args.splitfile, "w", encoding="utf-8") as fd:
        for idx in range(len(encodings.input_ids)):
            out = encodings['input_ids'][idx]
            try:
                start_idx = out.index(tokenizer.cls_token_id) + 1
                end_idx = out.index(tokenizer.sep_token_id) if (tokenizer.sep_token_id in out) else len(out)
                out_str = tokenizer.decode(out[start_idx:end_idx], skip_special_tokens=False)
                fd.write(out_str + "\n")
            except:
                print("Caught exception (for debugging).")
            # out_str = out_str.replace(tokenizer.pad_token, tokenizer.sep_token)
            # out_list = out_str.split(" ")
            # start_idx = out_list.index(tokenizer.cls_token) + 1
            # end_idx = out_list.index(tokenizer.sep_token) if tokenizer.sep_token in out_list else len(out_list)
            # fd.write(" ".join(out_list[start_idx:end_idx]) + "\n")

    print("Done Dataset Tokenizing")

