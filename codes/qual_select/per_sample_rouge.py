import argparse
import time
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
    parser.add_argument("--scorefile", type=str, required=True, help="File")

    parser.add_argument("--rouge_type", type=str, default="r2", help="Rouge Type (r2 is default)",
                        choices=["r1", "r2", "rL"])
    parser.add_argument("--pred_ct", type=int, default=1, help="Number of pred matching to each grd (must be constant)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Rouge alpha hyperparam"
                                                                 "(Precision vs Recall)")
    args = parser.parse_args()
    print("Get all Arguments:")

    grounds, predictions = get_dataset(args.grdfile, args.predfile, pred_ct=args.pred_ct)  # lock pred_Ct to 1

    all_rouge = []
    for idx, (grd, pred) in enumerate(zip(grounds, predictions)):
        reference = [grd.split()]
        hypothesis = pred.split()
        if args.rouge_type == "r1":
            all_rouge.append(rougescore.rouge_1(peer=hypothesis,
                                                models=reference,
                                                alpha=args.alpha))
        elif args.rouge_type == "r2":
            all_rouge.append(rougescore.rouge_2(peer=hypothesis,
                                                models=reference,
                                                alpha=args.alpha))
        elif args.rouge_type == "rL":
            all_rouge.append(rougescore.rouge_l(peer=hypothesis,
                                                models=reference,
                                                alpha=args.alpha))
        print("Sent Idx {}, Rouge Type {}, Rouge Score {}".format(idx, args.rouge_type, all_rouge[-1]))

    with open(args.scorefile, "w", encoding="utf-8") as fdout:
        for rs in all_rouge:
            fdout.write("{}\n".format(rs))
    print("Done")
