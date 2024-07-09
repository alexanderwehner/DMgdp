import torch
from torch import nn
from torch.nn import Linear, Sequential, ReLU
from transformers import XLMRobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class MlpParsingModel(nn.Module):
    def __init__(self, roberta_hidden_dim=768, mlp_dim=500, roberta_id="xlm-roberta-base", dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.roberta_hidden_dim = roberta_hidden_dim
        self.mlp_dim = mlp_dim

        # https://huggingface.co/docs/transformers/v4.34.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        self.roberta = XLMRobertaModel.from_pretrained(roberta_id)
        self.mlp = Sequential(Linear(roberta_hidden_dim, mlp_dim), ReLU(), Linear(mlp_dim, mlp_dim), ReLU())

        # freeze Roberta parameters
        for name, param in self.named_parameters():
            if name.startswith("roberta"):
                param.requires_grad = False

    def forward(self, x, attention_mask):
        # x: LongTensor (bs, seqlen)
        # attention_mask: IntTensor (bs, seqlen); 1 = real token, 0 = padding
        bs = x.shape[0]
        seqlen = x.shape[1]

        # encode x with Roberta
        out = self.roberta(x, attention_mask=attention_mask)
        hidden_states = out.last_hidden_state  # (bs, seqlen, hidden_size)

        emb = self.mlp(hidden_states)  # (bs, seqlen, mlp_dim)
        assert emb.shape == (bs, seqlen, self.mlp_dim)

        scores = torch.einsum("ijk,ilk->ijl", emb, emb)
        assert scores.shape == (bs, seqlen, seqlen)

        return scores


class DozatManningParsingModel(nn.Module):
    def __init__(self, roberta_hidden_dim=768, mlp_dim=500, roberta_id="xlm-roberta-base", dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.roberta_hidden_dim = roberta_hidden_dim
        self.mlp_dim = mlp_dim
        if activation == "relu":
            self.activation = nn.ReLU()

        # load XLM-RoBERTa
        self.roberta = XLMRobertaModel.from_pretrained(roberta_id)

        # define MLPs for head and dependent representations
        self.mlp_head = nn.Sequential(
            nn.Linear(roberta_hidden_dim, mlp_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim),
        )

        self.mlp_dep = nn.Sequential(
            nn.Linear(roberta_hidden_dim, mlp_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim),
        )

        # define parameters U1 and u2
        self.U1 = nn.Parameter(torch.Tensor(mlp_dim, mlp_dim))
        self.u2 = nn.Parameter(torch.Tensor(mlp_dim))

        # initialize parameters
        nn.init.xavier_uniform_(self.U1)
        nn.init.xavier_uniform_(self.u2.unsqueeze(0))

        # freeze Roberta parameters
        for name, param in self.named_parameters():
            if name.startswith("roberta"):
                param.requires_grad = False

    def forward(self, x, attention_mask):
        # x: LongTensor (bs, seqlen)
        # attention_mask: IntTensor (bs, seqlen); 1 = real token, 0 = padding
        bs, seqlen = x.shape

        # get embeddings
        roberta_output = self.roberta(x, attention_mask=attention_mask)
        hidden_states = roberta_output.last_hidden_state  # (bs, seqlen, hidden_size)

        # compute H_head and H_dep
        H_head = self.mlp_head(hidden_states)  # (bs, seqlen, mlp_dim)
        H_dep = self.mlp_dep(hidden_states)  # (bs, seqlen, mlp_dim)

        # compute scores
        scores = torch.einsum('bik,kl,bjl->bij', H_head, self.U1, H_dep)  # (bs, seqlen, seqlen)
        scores += torch.einsum('bik,k->bi', H_head, self.u2).unsqueeze(2)  # (bs, seqlen, seqlen)

        return scores

