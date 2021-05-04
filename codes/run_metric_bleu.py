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
from nltk.translate.bleu_score import SmoothingFunction

from nltk.translate import bleu_score

### Retrieve load the dataset from the corresponding files
def get_dataset(grdfile, predfile, pred_ct):
    with open(grdfile, "r", encoding="utf-8") as fd:
        grounds_base = [line.rstrip() for line in fd.readlines()]
    grounds = []
    for ground in grounds_base:
        for _ in range(pred_ct):
            grounds.append(ground)

    with open(predfile, "r", encoding="utf-8") as fd:
        predictions = [line.rstrip() for line in fd.readlines()]

    assert len(grounds) == len(predictions), \
        "Data Fields (Files) have different lengths"

    return grounds, predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Arguments")
    parser.add_argument("--grdfile", type=str, required=True, help="File")
    parser.add_argument("--predfile", type=str, required=True, help="File")

    parser.add_argument("--pred_ct", type=int, default=1, help="Number of pred matching to each grd (must be constant)")
    parser.add_argument("--do_sent_bleu", action="store_true", default=False, help="Compute Sent BLEU (not the traditional corpus BLEU)")
    args = parser.parse_args()
    print("Get all Arguments:")

    grounds, predictions = get_dataset(args.grdfile, args.predfile, args.pred_ct)

    smoothing_function = SmoothingFunction().method4
    if args.do_sent_bleu:
        for idx, (grd, pred) in enumerate(zip(grounds, predictions)):
            reference = [grd.split()]
            hypothesis = pred.split()
            sent_multi_bleu_score = bleu_score.sentence_bleu(references=reference, hypothesis=hypothesis,
                                                             smoothing_function=smoothing_function)
            print("Sent Idx {}, Multi-BLEU {}".format(idx, sent_multi_bleu_score))
    else:
        references = [[grd.split()] for grd in grounds]  # each sample only has one reference
        hypothesis = [pred.split() for pred in predictions]  # each sample only has one reference
        multi_bleu = bleu_score.corpus_bleu(list_of_references=references, hypotheses=hypothesis,
                                            smoothing_function=smoothing_function)
        print("Multi-BLEU {}".format(multi_bleu))

    print("Done")
