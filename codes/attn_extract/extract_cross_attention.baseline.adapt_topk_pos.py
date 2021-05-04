from transformers import AutoTokenizer, EncoderDecoderModel
import torch
import argparse
from zhon import hanzi
import numpy as np
import time


### Retrieve load the dataset from the corresponding files
def get_dataset(contentfile, queryfile):
    with open(queryfile, "r", encoding="utf-8") as fd:
        querys = [line.rstrip() for line in fd.readlines()]

    with open(contentfile, "r", encoding="utf-8") as fd:
        contents = [line.rstrip() for line in fd.readlines()]

    assert len(querys) == len(contents), \
        "Data Fields (Files) have different lengths"

    return contents, querys


def JustInTime_InOrder_Iterator(contents, querys, batch_size):  # torch dataset object
    assert len(contents) == len(querys), "Keywords and Querys are not the same length"
    batch = []
    for idx, (content, query) in enumerate(zip(contents, querys)):
        batch.append([content, query])
        if len(batch) == batch_size:
            yield [list(x) for x in list(zip(*batch))]  # tranpose to c, q
            batch = []
    if batch:  # clean out the leftovers
        yield [list(x) for x in list(zip(*batch))]  # tranpose to c, q


def args_parse():
    parser = argparse.ArgumentParser(description="Parse Command Line Arguments")
    parser.add_argument("--contentfile", type=str, required=True, help="Content File (Preferably Pretokenized for Consistency)")
    parser.add_argument("--queryfile", type=str, required=True, help="Query File (Preferably Pretokenized for Consistency)")

    parser.add_argument("--modelname", type=str, required=True, help="Model Folder")
    parser.add_argument("--outfile", type=str, required=True, help="Reduced Content Output File")
    parser.add_argument("--boundfile", type=str, required=True, help="Index Bound of Extracted Sentence File (Ignores [CLS] and pSEP] tokens)")

    parser.add_argument("--topk_pos_ratio", type=float, default=0.05, help="Ratio of Top Attentional Position to Extract")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")
    parser.add_argument("--sent_sep", action="store_true", default=False, help="Mark Sent Separation. If is False, sentences will be directly joined without separator")
    parser.add_argument("--do_viz", action="store_true", default=False, help="Do Visiualization, with Manual Iteration, for Demo")

    args = parser.parse_args()
    print("Get all Arguments:")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    return args


if __name__ == "__main__":
    print("Start Cross Attention Distribution Visualization", flush=True)
    args = args_parse()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]"]})
    model = EncoderDecoderModel.from_pretrained(args.modelname, output_attentions=True)
    model.encoder.config.output_attentions = True
    model.decoder.config.output_attentions = True
    model.to(device)
    model.eval()

    train_contents, train_querys, = get_dataset(args.contentfile,
                                                args.queryfile)

    print("Get Zeroth Sample", flush=True)
    print("Content: {}".format(train_contents[0]), flush=True)
    print("Question (Query): {}".format(train_querys[0]), flush=True)

    data_iterator = JustInTime_InOrder_Iterator(train_contents, train_querys,
                                                batch_size=args.batch_size)
    idx_sample = 0
    start = time.time()
    with open(args.outfile, "w", encoding="utf-8") as fdo, \
            open(args.boundfile, "w", encoding="utf-8") as fdb:
        with torch.no_grad():
            for global_idx, (contents, questions) in enumerate(data_iterator):  # one sample at a time
                inputs = tokenizer(contents, truncation=True, padding=True, return_tensors="pt")
                outputs = tokenizer(questions, truncation=True, padding=True, return_tensors="pt")
                input_ids = inputs.input_ids.to(device)
                input_mask = inputs.attention_mask.to(device)
                output_ids = outputs.input_ids.to(device)
                output_mask = outputs.attention_mask.to(device)
                model_outputs = model(input_ids=input_ids,
                                      attention_mask=input_mask,
                                      decoder_input_ids=output_ids,
                                      decoder_attention_mask=output_mask)
                attn_finlayers = model_outputs.cross_attentions[-1]
                attn_finlayers = attn_finlayers.mean(dim=1)  # aggregate by mean-pool across the heads (like channels)

                for local_idx in range(attn_finlayers.size(0)):
                    tokens_in = tokenizer.convert_ids_to_tokens(inputs.input_ids[local_idx])
                    tokens_out = tokenizer.convert_ids_to_tokens(outputs.input_ids[local_idx])
                    sep_ct_term = tokens_in.index("[SEP]")
                    tokens_ct = tokens_in[1:sep_ct_term]  # only get contents between CLS and SEP
                    sep_qu_term = tokens_out.index("[SEP]")
                    tokens_qu = tokens_out[:sep_qu_term]  # include CLS, as initial token still need to attend to content

                    # Should get position index of each sentence (start, end), to aggregate attentions
                    # TODO: How to slice index to include puntuation with sentence?
                    stop_tokens_ct_pos = np.array(  # Do consider using more "sentence markers"
                        [-1] + [i for i, x, in enumerate(tokens_ct) if x in hanzi.stops] + [len(tokens_ct)-1]) + 1
                    if stop_tokens_ct_pos[-1] == stop_tokens_ct_pos[-2]: # repeated end index
                        stop_tokens_ct_pos = stop_tokens_ct_pos[:-1]
                    stop_tokens_ct_range = np.array(
                        [*zip(stop_tokens_ct_pos[:], stop_tokens_ct_pos[1:])])  # range_min <= x <= range_max

                    attn_qu2ct = attn_finlayers[local_idx, :sep_qu_term, 1:sep_ct_term]  # retrieve query-encode attention (query may need to remove special characters)
                    attn_qu2ct_yaggr = attn_qu2ct.mean(dim=0)  # no longer a distribution (but that is fine)
                    assert attn_qu2ct_yaggr.shape[0] == len(tokens_ct)

                    key_pos_count = max(int(args.topk_pos_ratio * len(attn_qu2ct_yaggr)), 1) # ensure at least 1 position is retrieved
                    pos_values, pos_indices = torch.topk(attn_qu2ct_yaggr, k=key_pos_count)

                    bounds_set = set()
                    for pos_idx_ in pos_indices:
                        pos_idx = pos_idx_.item()
                        bound_a = stop_tokens_ct_range[(stop_tokens_ct_range[:, 0] <= pos_idx)][-1]
                        bound_b = stop_tokens_ct_range[(pos_idx < stop_tokens_ct_range[:, 1])][0]
                        assert np.all(bound_a == bound_b), "The selected bounds {} and {} does not match for sample {}".format(
                            bound_a, bound_b, global_idx + local_idx
                        )
                        bounds_set.add(tuple(bound_a))
                    bounds_selected_list = sorted(bounds_set, key=lambda x: x[0])

                    bounds_selection_label = [(bound[0], bound[1], int(tuple(bound) in bounds_selected_list)) for bound in stop_tokens_ct_range]

                    str_per_sent = [" ".join(tokens_ct[st: fi]) for st, fi in bounds_selected_list]
                    if args.sent_sep:
                        retrieved_sents = " [unused1] ".join(str_per_sent) + "\n"
                    else:
                        retrieved_sents = " ".join(str_per_sent) + "\n"
                    fdo.write(retrieved_sents)
                    bounds_str = ";".join(["{},{},{}".format(*ele) for ele in bounds_selection_label]) + "\n"
                    fdb.write(bounds_str)

                    # print(attended_sents, flush=True)
                    if ((idx_sample + 1) % 100 * args.batch_size) == 0:
                        end = time.time()
                        print("Done {} Samples, Time {}".format(idx_sample + 1, end - start), flush=True)
                    idx_sample += 1

                    if args.do_viz:
                        import matplotlib.pyplot as plt
                        from matplotlib import font_manager as fm

                        attn_qu2ct = attn_qu2ct.cpu()
                        attn_qu2ct_yaggr = attn_qu2ct_yaggr.cpu()

                        # display the results using matplotlib
                        fontP = fm.FontProperties(fname='c:\\windows\\fonts\\simsun.ttc')
                        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(30, 3), squeeze=False)
                        img0 = ax[0][0].imshow(attn_qu2ct, cmap='hot', interpolation='nearest', aspect="auto")
                        ax[0][0].set_xticks(np.arange(len(tokens_ct)))
                        ax[0][0].set_yticks(np.arange(len(tokens_qu)))
                        ax[0][0].set_xticklabels(tokens_ct, fontproperties=fontP, Fontsize=10)
                        ax[0][0].set_yticklabels(tokens_qu, fontproperties=fontP, Fontsize=10)
                        ax[0][0].set_title("Query-Content Encoder Attention for Huggingface Bert2Bert (Row is not Complete Distribution): Sample {}".format(global_idx+local_idx))

                        img1 = ax[1][0].imshow(attn_qu2ct_yaggr.unsqueeze(0), cmap='hot', aspect="auto")
                        ax[1][0].set_xticks(np.arange(len(tokens_ct)))
                        ax[1][0].set_xticklabels(tokens_ct, fontproperties=fontP, Fontsize=10)
                        ax[1][0].set_title("Query-Aggregated Encoder Content Attention for Huggingface Bert2Bert (Row is not Complete Distribution): Sample {}".format(global_idx+local_idx))
                        plt.show()

                        fig, ax = plt.subplots(nrows=3, ncols=1, squeeze=False)
                        ax[0][0].hist(attn_qu2ct_yaggr.numpy(), bins=50, rwidth=0.8)
                        ax[0][0].set_title("Pseudo (Unnormalized) PDF Histogram of Token Cross Attentions: Sample {}".format(global_idx+local_idx))
                        ax[0][0].set_xlabel("Attention Score")
                        ax[0][0].set_xlabel("Token/Position Count")

                        ax[1][0].hist(attn_qu2ct_yaggr.numpy(), bins=50, cumulative=True, rwidth=0.8)
                        ax[1][0].set_title("Pseudo (Unnormalized) CDF Histogram (X <= Thresh) of Token Cross Attentions: Sample {}".format(global_idx+local_idx))
                        ax[1][0].set_xlabel("Attention Score")
                        ax[1][0].set_xlabel("Token/Position Count")

                        ax[2][0].hist(attn_qu2ct_yaggr.numpy(), bins=50, cumulative=-1, rwidth=0.8)
                        ax[2][0].set_title("Pseudo (Unnormalized) Reversed CDF (X >= Thresh) Histogram of Token Cross Attentions: Sample {}".format(global_idx+local_idx))
                        ax[2][0].set_xlabel("Attention Score")
                        ax[2][0].set_xlabel("Token/Position Count")
                        plt.tight_layout()
                        plt.show()

    end = time.time()
    print("Total time: {}".format(end - start))

    print("Done")
