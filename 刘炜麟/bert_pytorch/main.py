import argparse

from torch.utils.data import DataLoader

from model import BERT
from trainer import BERTTrainer
from dataset import BERTDataset, WordVocab


def train(args):
    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Building BERT model")
    bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

    print("Creating BERT Trainer")
    trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__=="__main__":
    class Args():
        def __init__(self):
            self.train_dataset="data/corpus.small"      #train dataset for train bert
            self.test_dataset=None                      #test set for evaluate train set, default=None
            self.vocab_path="data/vocab.small"          #built vocab model path with bert-vocab
            self.output_path="output/bert.model"        #ex)output/bert.model
            self.hidden=256                             #hidden size of transformer model, default=256
            self.layers=8                               #number of layers, default=8
            self.attn_heads=8                           #number of attention heads, default=8
            self.seq_len=20                             #maximum sequence len, default=20
            self.batch_size=64                          #number of batch_size, default=64
            self.epochs=10                              #number of epochs, default=10
            self.num_workers=1                          #dataloader worker size, default=5
            self.with_cuda=False                        #training with CUDA: true, or false, default=True
            self.log_freq=10                            #printing loss every n iter: setting n, default=10
            self.corpus_lines=None                      #total number of lines in corpus, default=None
            self.cuda_devices=None                      #CUDA device ids, default=None
            self.on_memory=True                         #Loading on memory: true or false, default=True
            self.lr=1e-3                                #learning rate of adam, default=1e-3
            self.adam_weight_decay=0.01                 #weight_decay of adam, default=0.01
            self.adam_beta1=0.9                         #adam first beta value, default=0.9
            self.adam_beta2=0.999                       #adam first beta value, default=0.999
    
    args=Args()
    train(args)