# encoding: utf-8
"""
@author:  Remi Lebret
@contact: remi@lebret.ch
"""
import torch
import torch.nn as nn

from transformers import BertConfig, BertModel, BertTokenizer
from lavis.models import load_model


class TokenAlignmentModel(nn.Module):
    def __init__(
        self,
        teacher_model_name: str = "albef_pretrain",
        teacher_model_type: str = "flickr",
        student_model_name: str = "bert-base-multilingual-cased",
        num_layers: int = 1,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.teacher_model = load_model(
            name=teacher_model_name,
            model_type=teacher_model_type,
            is_eval=True,
            device=device,
        )

        student_config = BertConfig.from_pretrained(student_model_name)
        student_config.num_hidden_layers = 6
        self.student_model = self.create_student_model(student_model_name)

        self.embedding_size = student_config.hidden_size

    def train(self, mode: bool = True):
        self.teacher_model.eval()
        self.student_model.train()

    def forward(self, inputs):
        (
            src_input_ids,
            tgt_input_ids,
            src_attention_mask,
            tgt_attention_mask,
            src_idx,
            tgt_idx,
            _,
        ) = inputs
        assert len(src_idx) == len(tgt_idx)
        num_token = len(src_idx)
        device = src_input_ids.device

        with torch.no_grad():
            src_outputs = self.teacher_model.text_encoder(
                src_input_ids,
                attention_mask=src_attention_mask,
                output_hidden_states=True,
                mode="text",
            )
        tgt_outputs = self.student_model(
            tgt_input_ids, attention_mask=tgt_attention_mask, output_hidden_states=True
        )

        src_token_hidden_states = torch.zeros(
            self.num_layers, num_token, self.embedding_size, device=device
        )
        tgt_token_hidden_states = torch.zeros(
            self.num_layers, num_token, self.embedding_size, device=device
        )
        for i, layer in enumerate(range(-1, -1 - self.num_layers, -1)):
            src_token_hidden_states[i] = torch.index_select(
                src_outputs.hidden_states[layer].view(-1, self.embedding_size),
                0,
                src_idx,
            )
            tgt_token_hidden_states[i] = torch.index_select(
                tgt_outputs.hidden_states[layer].view(-1, self.embedding_size),
                0,
                tgt_idx,
            )

        return src_token_hidden_states, tgt_token_hidden_states

    @staticmethod
    def create_student_model(student_model_name):
        student_config = BertConfig.from_pretrained(student_model_name)
        student_config.num_hidden_layers = 6
        return BertModel.from_pretrained(
            student_model_name, config=student_config, add_pooling_layer=False
        )

    @staticmethod
    def init_from_student_model(
        model, state_dict, student_model_name="bert-base-multilingual-cased"
    ):
        student_model = TokenAlignmentModel.create_student_model(student_model_name)
        student_state_dict = {
            k.replace("student_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("student_model.")
        }
        student_model.load_state_dict(student_state_dict)
        student_model.to(model.device)
        model.text_encoder.embeddings = (
            student_model.embeddings
        )  # replace the embeddings entirely
        for i in range(
            len(student_model.encoder.layer)
        ):  # replace only the encoder layers parameters
            model.text_encoder.encoder.layer[i].load_state_dict(
                student_model.encoder.layer[i].state_dict()
            )
        # replace the model tokenizer with the multilingual text encoder tokenizer
        model.tokenizer = BertTokenizer.from_pretrained(student_model_name)
