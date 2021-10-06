import hanlp
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH) # 世界最大中文语料库
a='2021年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。'
c='2021年HanLPv2.1为实验环境带来次世代最先进的多语种NLP技术。'
b=HanLP(a)
lenvocab=0
vocab={}


def setencetodict(sentence:str,vocab:dict,lenvocab:int):
    #输入句子分词后生成词典
    #输入：sentence:str,vocab:dict,lenvocab:int
    #返回：
    splitsentence:dict=HanLP(sentence)['tok/fine']
    print(splitsentence)
    for i in splitsentence:
        if vocab.__contains__(i):
            pass
        else:
            vocab[i]=lenvocab
            lenvocab+=1
    return lenvocab,vocab

lenvocab,vocab=setencetodict(c,vocab,lenvocab)
lenvocab,vocab=setencetodict(a,vocab,lenvocab)

index2word=dict(zip(vocab.values(),vocab.keys()))
word2index=vocab

print(index2word)
print(word2index)
'''

def sen_to_split_emotion(sentence:str)->list: #输入句子 返回 切割与词性  使用pos/863标准
    splitsentence=[[],[]]
    splitsentence[0]=HanLP(sentence)['tok/fine']
    splitsentence[1]=HanLP(sentence)['pos/863']
    return splitsentence

pos_all=["ng","nt","nd","nl","nh","ns","nn","ni","nz","vt","vi","vl","vu","vd","aq","as","in","iv","ia","ic","jn",'"jv"',"ja","gn","gv","ga","wp","ws","wu","n","w","p","v","a","d","u"]
pos2index = {pos: i for i, pos in enumerate(pos_all)}
index2pos = {pos2index[pos]: pos for pos in pos2index}

'''

pos_tag_ids_map = {'u':0, 'v':1, 'a':2, 'r':3, 'n':4}
def convert_postag(pos):
    """Convert NLTK POS tags to SWN's POS tags."""
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

HanLP(a)['pos/ctb']