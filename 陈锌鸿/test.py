from config import config
from uer.targets.psm_target import PsmTarget
from uer.targets.bert_target import *
import numpy as np


if __name__ == "__main__":
    config = config()
    '''
    memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]
    '''

    memory_bank = torch.randn([config.batch_size,config.max_seq_length,config.hidden_size],dtype=torch.long)
    tgt_psm = torch.randn([config.batch_size,config.max_seq_length],dtype=torch.long)

    print(type(memory_bank))
    PSM = PsmTarget(config,config.vocab_size)
    loss_psm, correct_psm, denominator = PSM.psm(memory_bank, tgt_psm)
    print(loss_psm, correct_psm, denominator)
