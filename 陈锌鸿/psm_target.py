import torch
import torch.nn as nn
from uer.layers.layer_norm import LayerNorm
from uer.utils import *


class PsmTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    and Part of speech model (PSM).
    """

    def __init__(self, args, vocab_size):
        super(PsmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.act = str2act[args.hidden_act]

        if self.factorized_embedding_parameterization:
            self.psm_linear_1 = nn.Linear(args.hidden_size, args.emb_size)
            self.layer_norm = LayerNorm(args.emb_size)
            self.psm_linear_2 = nn.Linear(args.emb_size, self.vocab_size)
        else:
            self.psm_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
            self.layer_norm = LayerNorm(args.hidden_size)
            self.psm_linear_2 = nn.Linear(args.hidden_size, self.vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)

        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()

    def psm(self, memory_bank, tgt_psm):
        #  Part of speech modeling (psm) with full softmax prediction.
        output_psm = self.act(self.psm_linear_1(memory_bank))
        output_psm = self.layer_norm(output_psm)

        if self.factorized_embedding_parameterization:
            output_psm = output_psm.contiguous().view(-1, self.emb_size)
        else:
            output_psm = output_psm.contiguous().view(-1, self.hidden_size)
        tgt_psm = tgt_psm.contiguous().view(-1)
        output_psm = output_psm[tgt_psm > 0, :]
        tgt_psm = tgt_psm[tgt_psm > 0]
        output_psm = self.psm_linear_2(output_psm)
        output_psm = self.softmax(output_psm)
        denominator = torch.tensor(output_psm.size(0) + 1e-6)
        if output_psm.size(0) == 0:
            correct_psm = torch.tensor(0.0)
        else:
            correct_psm = torch.sum((output_psm.argmax(dim=-1).eq(tgt_psm)).float())

        loss_psm = self.criterion(output_psm, tgt_psm)
        return loss_psm, correct_psm, denominator

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Part of speech modeling loss.
            correct: Number of Part of speech that are predicted correctly.
            denominator: Number of masked Part of speech words.
        """

        # Part of speech model (psm).
        loss, correct, denominator = self.psm(memory_bank, tgt)

        return loss, correct, denominator

