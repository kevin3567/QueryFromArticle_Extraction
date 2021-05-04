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


### Retrieve load the dataset from the corresponding files
def get_dataset(contentfile):
    with open(contentfile, "r", encoding="utf-8") as fd:
        contents = [line.rstrip() for line in fd.readlines()]

    return contents


def show_answer(tokenizer_model, in_encodings, idx):
    print("Index :: ", idx)
    print("Input (Content) :: ", tokenizer_model.decode(in_encodings['input_ids'][idx]))
    print("")


class GenerationDataset(torch.utils.data.Dataset):  # torch dataset object
    def __init__(self, input_encodings):  # takes in encoded datasets generated by huggingface tokenizers
        self.input_encodings = input_encodings

    def __getitem__(self, idx):  # for each idx, pass the input encodings and output encodings
        input_dict = {key: torch.tensor(val[idx]) for key, val in self.input_encodings.items()}
        return input_dict

    def __len__(self):  # get the length (number of samples) in the dataset
        return len(self.input_encodings.input_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferring Arguments")
    parser.add_argument("--contentfile", type=str, required=True, help="Content File (Input)")
    parser.add_argument("--modelckpt", type=str, required=True, help="Model Checkpoint (To Load)")
    parser.add_argument("--outfile", type=str, required=True, help="Query File (Output)")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch Size")

    args = parser.parse_args()
    print("Get all Arguments:")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    test_contents = get_dataset(args.contentfile)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]"]})
    test_input_encodings = tokenizer(test_contents, truncation=True, padding=True)

    print("Show some examples: ")
    show_answer(tokenizer, test_input_encodings, 0)
    # show_answer(tokenizer, test_input_encodings, 100)
    # show_answer(tokenizer, test_input_encodings, 2000)
    show_answer(tokenizer, test_input_encodings, -1)
    print("Length of Infer Set: {}".format(len(test_contents)))
    print("Done Dataset Processing")

    test_dataset = GenerationDataset(test_input_encodings)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    """Multiple Instance Inference"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = EncoderDecoderModel.from_pretrained(args.modelckpt,
                                                output_attentions=True,
                                                output_hidden_states=True)
    model.to(device)
    model.eval()
    with torch.no_grad():
        fd = open(args.outfile, "w", encoding="utf-8")
        start = time.time()
        for step, batch_in in enumerate(test_loader):
            input_ids = batch_in["input_ids"].to(device)
            attention_mask = batch_in["attention_mask"].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     decoder_start_token_id=tokenizer.cls_token_id, eos_token_id=tokenizer.sep_token_id,
                                     num_beams=5, num_return_sequences=1,
                                     min_length=3, max_length=15)
            output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            for index, out in enumerate(outputs):
                try:
                    outlist = out.tolist()
                    start_idx = outlist.index(tokenizer.cls_token_id) + 1
                    end_idx = outlist.index(tokenizer.sep_token_id) if (tokenizer.sep_token_id in out) else len(out)
                    out_str = tokenizer.decode(out[start_idx:end_idx], skip_special_tokens=False)
                    fd.write(out_str + "\n")
                except:
                    print("Caught exception (for debugging).")
            if ((step + 1) % 200) == 0:
                end = time.time()
                print("Step: {}, Duration: {}".format(step + 1, end - start), flush=True)
                start = time.time()
        end = time.time()
        print("Step: {}, Duration: {}".format(step + 1, end - start))
        fd.close()

    print("Done")
