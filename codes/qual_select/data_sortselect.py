import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="Based on score, retrieve top-K samples from corresponding infiles.")

parser.add_argument("--scorefile", type=str, required=True, help="Sample scores")
parser.add_argument("--infiles", type=str, required=True, nargs="+", help="Input files to retrieve samples from")
parser.add_argument("--outtag", type=str, required=True, help="Output files tag (append at the end) to store selected samples in")
parser.add_argument("--outdir", type=str, required=True, help="Output directory for output files")

parser.add_argument("--score_thresh", type=float, default=0.5, help="Score_threshold to filter out samples")

args = parser.parse_args()

print("Do path check of infiles")
for f in args.infiles:
    assert os.path.isfile(f), "{} does not exist (or is not a file)".format(f)
if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)

print("Sort index by score")
with open(args.scorefile, "r", encoding="utf-8") as fdsc:
    scores = [float(sc) for sc in fdsc]
    indices = list(range(len(scores)))
    idx_sc_tuples = list(zip(scores, indices))
    idx_sc_tuples_sorted = sorted(idx_sc_tuples, key=lambda x: x[0], reverse=True)

print("Record selected indices")
sel_idx = []
idxfile = os.path.join(args.outdir, os.path.basename(args.scorefile)+".scth_{}.sel_idx".format(args.score_thresh*100))  # TODO: resolve same filename-collisions
with open(idxfile, "w", encoding="utf-8") as fdidx:
    for sc, idx in idx_sc_tuples_sorted:
        if sc >= args.score_thresh:
            fdidx.write("{}, {}\n".format(idx, sc))
            sel_idx.append(idx)

print("Begin filtering samples from infiles")
for f in args.infiles:
    print("Do {}".format(f))
    with open(f, "r", encoding="utf-8") as fd:
        lines = fd.readlines()
        assert len(lines) == len(idx_sc_tuples_sorted), "Infiles does not have same number of samples as that of Scorefile"
    outf = os.path.join(args.outdir, os.path.basename(f)+".scth_{}.{}".format(args.score_thresh*100, args.outtag))  # TODO: resolve same filename-collisions
    with open(outf, "w", encoding="utf-8") as fd:
        for idx in sel_idx:
            fd.write(str(lines[idx]))
    print("Done {}".format(f))


print("Done")

