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
# import metrics.rougescore_master.rougescore.rougescore as rougescore
import rougescore.rougescore as rougescore


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
    parser.add_argument("--alpha", type=float, default=0.5, help="Rouge alpha hyperparam"
                                                                 "(Precision vs Recall)")
    args = parser.parse_args()
    print("Get all Arguments:")

    grounds, predictions = get_dataset(args.grdfile, args.predfile, pred_ct=args.pred_ct)  # lock pred_Ct to 1

    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []
    for idx, (grd, pred) in enumerate(zip(grounds, predictions)):
        reference = [grd.split()]
        hypothesis = pred.split()
        all_rouge1.append(rougescore.rouge_1(peer=hypothesis,
                                             models=reference,
                                             alpha=args.alpha))
        all_rouge2.append(rougescore.rouge_2(peer=hypothesis,
                                             models=reference,
                                             alpha=args.alpha))
        all_rougeL.append(rougescore.rouge_l(peer=hypothesis,
                                             models=reference,
                                             alpha=args.alpha))
        # print("Sent Idx {}, Rouge-1 {}, Rouge-2 {}, Rouge-3 {}".format(idx, all_rouge1[-1], all_rouge2[-1],
        #                                                                all_rougeL[-1]))

    print("Total Averaged Rouge: Rouge-1 {}, Rouge-2 {}, Rouge-3 {}".format(np.mean(all_rouge1),
                                                                            np.mean(all_rouge2),
                                                                            np.mean(all_rougeL)))

    print("Done")
