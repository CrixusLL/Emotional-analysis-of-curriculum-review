from dataset import vocab

class Args:
    def __init__(self):
        self.corpus_path="data/corpus.small"
        self.output_path="data/vocab.small"
        self.vocab_size=None
        self.encoding="utf-8"
        self.min_freq=1


if __name__=="__main__":
    args=Args()
    vocab.build(args)