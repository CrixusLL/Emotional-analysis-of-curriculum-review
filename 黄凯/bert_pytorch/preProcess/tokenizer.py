import os
import json
import hanlp

HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)

class Tokenizer(object):
    def __init__(
        self,
        create_wordDict: bool = True,
        data_directory_path: str = None,
        filePath_word_to_idx: str = None,
        filePath_idx_to_word: str = None
    ):
        super().__init__()
        self.segment = HanLP
        self.unk_token="[UNK]"
        self.sep_token="[SEP]"
        self.pad_token="[PAD]"
        self.cls_token="[CLS]"
        self.mask_token="[MASK]"
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

        if create_wordDict:
            self.data_paths = self.findAllFile(data_directory_path)
            self.documents = self.loadData(self.data_paths)
            self.word_to_idx, self.idx_to_word = self.create_word_dict()
            self.save_word_dict(save_filePath_word_to_idx=filePath_word_to_idx, save_filePath_idx_to_word=filePath_idx_to_word)
        else:
            self.load_word_dict(load_filePath_word_to_idx=filePath_word_to_idx, load_filePath_idx_to_word=filePath_idx_to_word)
        self.pos_to_idx, self.idx_to_pos = self.create_pos_dict()

    @staticmethod
    def findAllFile(data_path):
        """
        返回可迭代对象（文本路径）
        """
        for root, ds, fs in os.walk(data_path):
            for f in fs:
                fullname = os.path.join(root, f)
                yield fullname

    @staticmethod
    def loadData(paths):
        documents = []
        for path in paths:
            f = open(path)
            s = f.readlines()
            for line in s:
                document = []
                for sentence in json.loads(line)["text"].split('\n'):
                    if sentence:
                        document.append(sentence)
                documents.append(document)
        return documents

    def create_word_dict(self):
        """构建词字典"""
        all_sentences = [sentence for document in self.documents for sentence in document]
        words_list = [self.segment(sentence)["tok/fine"] for sentence in all_sentences]
        vocb = list(set([word for words in words_list for word in words]))
        # 构建word2index字典（留出0,1,2,3,4作为"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"）
        len_specialTokens = len(self.special_tokens)
        word_to_idx = {(self.special_tokens[i] if i < len_specialTokens else vocb[i - len_specialTokens]):i for i in range(len(vocb) + len_specialTokens)}
        idx_to_word = {word_to_idx[word]: word for word in word_to_idx}
        return word_to_idx, idx_to_word

    @property
    def vocab_size(self):
        return len(self.word_to_idx)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list, token_ids_1: list = None
    ) -> list:
        """
        Build model inputs from a sequence or a pair of sequence for sequence 
        classification tasks by concatenating and adding special tokens. 
        A BERT sequence has the following format:
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        else:
            cls = [self.cls_token_id]
            sep = [self.sep_token_id]
            return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0: list, token_ids_1: list = None
    ) -> list:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        else:
            return [1] + ([0] * len(token_ids_0)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list, token_ids_1: list = None
    ) -> list:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. A BERT sequence
        pair mask has the following format:
        ::
            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |
        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `token type IDs <../glossary.html#token-type-ids>`_ according to the given
            sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        else:
            return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def save_word_dict(self, save_filePath_word_to_idx: str, save_filePath_idx_to_word: str):
        json.dump(self.word_to_idx, open(save_filePath_word_to_idx, "w"), ensure_ascii=False)
        json.dump(self.idx_to_word, open(save_filePath_idx_to_word, "w"), ensure_ascii=False)

    def load_word_dict(self, load_filePath_word_to_idx: str, load_filePath_idx_to_word: str):
        with open(load_filePath_word_to_idx, "r", encoding='utf-8') as f:
            self.word_to_idx = json.loads(f.read())
        with open(load_filePath_idx_to_word, "r", encoding='utf-8') as f:
            self.idx_to_word = json.loads(f.read())
        
    def create_pos_dict(self):
        """构建词性字典"""
        pos_to_idx = {'u':0, 'v':1, 'a':2, 'r':3, 'n':4, '[MASK]':5}
        idx_to_pos = {pos_to_idx[pos]: pos for pos in pos_to_idx}
        return pos_to_idx, idx_to_pos

    @property
    def pos_tag_size(self):
        return len(self.pos_to_idx)

    @staticmethod
    def convert_pos_tag(pos):
        """转换词性至自定义大类"""
        if pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return 'v'
        elif pos in ['JJ', 'JJR', 'JJS']:
            return 'a'
        elif pos in ['RB', 'RBR', 'RBS']:
            return 'r'
        elif pos in ['NNS', 'NN', 'NNP', 'NNPS']:
            return 'n'
        else:
            return 'u'

    def convert_tokens_to_ids(self, tokens):
        output = []
        for item in tokens:
            output.append(self.word_to_idx[item])
        return output

    def convert_pos_tags_to_ids(self, pos_tags):
        output = []
        for item in pos_tags:
            output.append(self.pos_to_idx[item])
        return output

if __name__ == "__main__":
    # tokenizer = Tokenizer(
    #     create_wordDict=True,
    #     data_directory_path="./bert_pytorch/data/miniData_debug/",
    #     filePath_word_to_idx="./bert_pytorch/data/save/word_to_idx.json",
    #     filePath_idx_to_word="./bert_pytorch/data/save/idx_to_word.json"
    # )
    tokenizer = Tokenizer(
        create_wordDict=False,
        data_directory_path=None,
        filePath_word_to_idx="./bert_pytorch/data/save/word_to_idx.json",
        filePath_idx_to_word="./bert_pytorch/data/save/idx_to_word.json"
    )