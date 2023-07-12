import json
import jsonlines
import itertools
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def sentence_to_ids(sentence: str, tokenizer: AutoTokenizer):
    sent = sentence.strip().split()
    tokenized_sent = [tokenizer.tokenize(word) for word in sent]
    token_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_sent]
    tensor_ids = tokenizer.prepare_for_model(
        list(itertools.chain(*token_ids)),
        return_tensors="pt",
        model_max_length=tokenizer.model_max_length,
        truncation=True,
    )["input_ids"]
    sub2word_map = [i for i, word_list in enumerate(tokenized_sent) for x in word_list]
    return tensor_ids, sub2word_map


def get_word_pairs(src, tgt, model, tokenizer, device):
    """using awesome-align
    awesome-align is a tool that can extract word alignments
    from multilingual BERT (mBERT)
    https://arxiv.org/abs/2101.08231"""

    # pre-processing
    ids_src, sub2word_map_src = sentence_to_ids(src, tokenizer)
    ids_tgt, sub2word_map_tgt = sentence_to_ids(tgt, tokenizer)

    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0).to(device), output_hidden_states=True)[2][
            align_layer
        ][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0).to(device), output_hidden_states=True)[2][
            align_layer
        ][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
        align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

    return align_words


def load_sentences(annotations: List, item: str = "sentence"):
    sentences = [ann[item] for ann in annotations]

    # flatten list of lists if necessary
    def flatten_list(_2d_list):
        flat_list = []
        # Iterate through the outer list
        for element in _2d_list:
            if type(element) is list:
                # If the element is of type list, iterate through the sublist
                for item in element:
                    flat_list.append(item)
            else:
                flat_list.append(element)
        return flat_list

    return flatten_list(sentences)


def generate_token_pairs(
    annotation_src_path,
    annotation_tgt_path,
    output_path,
    item="sentence",
    device="cuda:0",
    model_name="aneuraz/awesome-align-with-co",
):
    sentences_src = load_sentences(jsonlines.open(annotation_src_path), item=item)
    sentences_tgt = load_sentences(jsonlines.open(annotation_tgt_path), item=item)
    assert len(sentences_src) == len(sentences_tgt), "number of sentences mismatch"
    n_sentences = len(sentences_src)

    # load models
    # for awesome-align, use "aneuraz/awesome-align-with-co"
    # for mBERT, use "bert-base-multilingual-cased"
    # both models shared the same tokenizer as awesome-align has been trained on mBERT
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # dataset_token_pairs = []
    dataset_word_pairs = []
    print("generating token pairs...")
    for i in tqdm(range(n_sentences)):
        sentence_src, sentence_tgt = sentences_src[i], sentences_tgt[i]
        word_pairs = get_word_pairs(
            sentence_src, sentence_tgt, model, tokenizer, device
        )
        dataset_word_pairs.append(list(word_pairs))

    print("saving token pairs...")
    with open(output_path, "w") as f:
        for i in tqdm(range(n_sentences)):
            obj = {
                "src": sentences_src[i],
                "tgt": sentences_tgt[i],
                "alignment": dataset_word_pairs[i],
            }
            json.dump(obj, f)
            f.write("\n")
