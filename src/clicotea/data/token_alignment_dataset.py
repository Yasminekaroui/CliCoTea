# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
import jsonlines
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def prepare_token_alignment_batch(batch, device):
    (
        src_tensor,
        tgt_tensor,
        src_attention_mask,
        tgt_attention_mask,
        src_idx,
        tgt_idx,
        num_pairs,
    ) = batch
    src_tensor = src_tensor.to(device)
    tgt_tensor = tgt_tensor.to(device)
    src_attention_mask = src_attention_mask.to(device)
    tgt_attention_mask = tgt_attention_mask.to(device)
    src_idx = src_idx.to(device)
    tgt_idx = tgt_idx.to(device)
    num_pairs = num_pairs.to(device)
    return (
        src_tensor,
        tgt_tensor,
        src_attention_mask,
        tgt_attention_mask,
        src_idx,
        tgt_idx,
        num_pairs,
    )


class TokenAlignmentDataset(Dataset):
    def __init__(
        self,
        token_alignment_file: str,
        src_tokenizer_name: str,
        tgt_tokenizer_name: str = "bert-base-multilingual-cased",
        align_with_cls: bool = False,
        add_en_sent: bool = False,
    ) -> None:
        super().__init__()
        self.token_alignment_file = token_alignment_file
        self.align_with_cls = align_with_cls
        self.src_tokenizer = AutoTokenizer.from_pretrained(src_tokenizer_name)
        if src_tokenizer_name == tgt_tokenizer_name:
            self.tgt_tokenizer = self.src_tokenizer
        else:
            self.tgt_tokenizer = AutoTokenizer.from_pretrained(tgt_tokenizer_name)

        # define padding ids
        self.src_padding_id = self.src_tokenizer.convert_tokens_to_ids(
            self.src_tokenizer.pad_token
        )
        self.tgt_padding_id = self.src_tokenizer.convert_tokens_to_ids(
            self.src_tokenizer.pad_token
        )
        # load input file
        self.samples = []
        for obj in jsonlines.open(self.token_alignment_file):
            src_sent = obj["src"].strip().split()
            tgt_sent = obj["tgt"].strip().split()
            self.samples.append((src_sent, tgt_sent, obj["alignment"]))
            if add_en_sent:
                self.samples.append(
                    (src_sent, src_sent, [[i, i] for i in range(len(src_sent))])
                )

    def collate_fn(self, batch):
        max_src = max([len(t["input_ids"]) for t, _, _ in batch])
        max_tgt = max([len(t["input_ids"]) for _, t, _ in batch])
        num_pairs = torch.tensor([len(p) for _, _, p in batch], dtype=torch.long)
        src_tensor = torch.zeros(len(batch), max_src, dtype=torch.long).fill_(
            self.src_padding_id
        )
        tgt_tensor = torch.zeros(len(batch), max_tgt, dtype=torch.long).fill_(
            self.tgt_padding_id
        )
        src_attention_mask = torch.zeros(len(batch), max_src, dtype=torch.long)
        tgt_attention_mask = torch.zeros(len(batch), max_tgt, dtype=torch.long)
        src_idx = []
        tgt_idx = []
        for i, (s, t, p) in enumerate(batch):
            src_tensor[i, : len(s["input_ids"])] = torch.tensor(
                s["input_ids"], dtype=torch.long
            )
            tgt_tensor[i, : len(t["input_ids"])] = torch.tensor(
                t["input_ids"], dtype=torch.long
            )
            src_attention_mask[i, : len(s["attention_mask"])] = torch.tensor(
                s["attention_mask"], dtype=torch.long
            )
            tgt_attention_mask[i, : len(t["attention_mask"])] = torch.tensor(
                t["attention_mask"], dtype=torch.long
            )
            src_idx += [a + i * max_src for a, _ in p]
            tgt_idx += [b + i * max_tgt for _, b in p]

        src_idx = torch.tensor(src_idx, dtype=torch.long)
        tgt_idx = torch.tensor(tgt_idx, dtype=torch.long)

        return (
            src_tensor,
            tgt_tensor,
            src_attention_mask,
            tgt_attention_mask,
            src_idx,
            tgt_idx,
            num_pairs,
        )

    def get_loader(self, batch_size, num_workers=0, train=False):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=train,
            collate_fn=self.collate_fn,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, align = self.samples[idx]
        src_tokens = self.src_tokenizer(src, is_split_into_words=True, truncation=True)
        tgt_tokens = self.tgt_tokenizer(tgt, is_split_into_words=True, truncation=True)
        src_word_ids = src_tokens.word_ids()
        tgt_word_ids = tgt_tokens.word_ids()
        word_pairs = [
            (src_word_ids.index(s), tgt_word_ids.index(t))  # take only left alignment
            for s, t in align
            if s in src_word_ids and t in tgt_word_ids
        ]
        if self.align_with_cls:
            word_pairs = [(0, 0)] + word_pairs
        return (src_tokens, tgt_tokens, word_pairs)
