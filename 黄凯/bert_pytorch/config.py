import torch

class config(object):
    def __init__(self):
        self.pos_tag_size = 6 # part of speech categories, pos_to_idx = {'u':0, 'v':1, 'a':2, 'r':3, 'n':4, '[MASK]':5}

        self.random_seed = 12345

        self.use_cuda = torch.cuda.is_available()

        self.max_seq_length = 128
        self.max_predictions_per_seq = 20
        self.dupe_factor = 10
        self.masked_lm_prob = 0.15
        self.short_seq_prob = 0.1

        self.vocab_size=30522
        self.hidden_size=768
        self.num_hidden_layers=12
        self.num_attention_heads=12
        self.intermediate_size=3072
        self.hidden_act="gelu"
        self.hidden_dropout_prob=0.1
        self.attention_probs_dropout_prob=0.1
        self.max_position_embeddings=512
        self.type_vocab_size=2
        self.initializer_range=0.02
        self.layer_norm_eps=1e-12
        self.pad_token_id=0
        self.position_embedding_type="absolute"
        self.use_cache=True
        self.classifier_dropout=None

        # vocab_size = 5000
        # maxlen = 30 # maximum of length
        # batch_size = 6
        # max_pred = 5  # max tokens of prediction
        # n_layers = 6 # number of Encoder of Encoder Layer
        # n_heads = 12 # number of heads in Multi-Head Attention
        # d_model = 768 # Embedding Size
        # d_ff = 768 * 4  # 4*d_model, FeedForward dimension
        # d_k = d_v = 64  # dimension of K(=Q), V
        # n_segments = 2
        # pos_tag_size = 5 # part of speech categories
        # pos_tags_pad_idx = 4
        # hidden_dropout_prob = 0.3
        # layer_norm_eps = 1e-12