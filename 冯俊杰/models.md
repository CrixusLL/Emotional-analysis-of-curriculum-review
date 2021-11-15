# Google research-base
link：https://github.com/google-research/bert

参数：
* BERT-large, Chinese：24-layer, 1024-hidden, 16-heads, 330M parameters
* BERT-base, Chinese：12-layer, 768-hidden, 12-heads, 110M parameters
  
常见基于BERT的中文预训练模型介绍：https://arxiv.org/pdf/2004.13922

# RoBERTa-zh
link：https://github.com/brightmart/roberta_zh

语料：使用30G中文训练，包含3亿个句子，100亿个字(即token）。由新闻、社区讨论、多个百科，包罗万象，覆盖数十万个主题，所以数据具有多样性（为了更有多样性，可以可以加入网络书籍、小说、故事类文学、微博等）

训练时间：总共训练了近20万，总共见过近16亿个训练数据(instance)； 在Cloud TPU v3-256 上训练了24小时，相当于在TPU v3-8(128G显存)上需要训练一个月

效果：![效果](https://image.jiqizhixin.com/uploads/editor/e7d27c6a-0e43-4cdc-b105-0e7c0046a91e/640.jpeg)

# Bert-wwm
link：https://github.com/ymcui/Chinese-BERT-wwm

效果：任务为基于二分类的情感分类数据集ChnSentiCorp的情感分析任务，评价指标为：Accuracy

| 模型 | 开发集 | 测试集 |
| ---- | ---- | ---- |
| BERT-base | 94.7 (94.3) | 95.0 (94.7)|
| ERNIE | 95.4 (94.8) | 95.4 (95.3)|
| BERT-wwm | 95.1 (94.5) | 95.4 (95.0)|
| BERT-wwm-ext | 95.4 (94.6) | 95.3 (94.7)|
| RoBERTa-wwm-ext | 95.0 (94.6) | 95.6 (94.8)|
| RoBERTa-wwm-ext-large | 95.8 (94.9) | 95.8 (94.9)|

# ELECTRA
link：https://github.com/ymcui/Chinese-ELECTRA

效果：任务为基于二分类的情感分类数据集ChnSentiCorp的情感分析任务，评价指标为：Accuracy

| 模型 | 开发集 | 测试集 | 参数量 |
| ---- | ---- | ---- | ---- |
| BERT-base | 94.7 (94.3) | 95.0 (94.7)| 102M |
| BERT-wwm | 95.1 (94.5) | 95.4 (95.0)| 102M |
| BERT-wwm-ext | 95.4 (94.6) | 95.3 (94.7)| 102M |
| RoBERTa-wwm-ext | 95.0 (94.6) | 95.6 (94.8)| 102M |
| RBT3 | 92.8 | 92.8| 38M |
| ELECTRA-small | 92.8 (92.5) | 94.3 (93.5)| 12M |
| ELECTRA-180g-small | 94.1 | 93.6| 12M |
| ELECTRA-small-ex | 92.6 | 93.6| 25M |
| ELECTRA-180g-small-ex | 92.8 | 93.4| 325M |
| ELECTRA-base | 93.8 (93.0) | 94.5 (93.5)| 102M |
| ELECTRA-180g-base | 94.3 | 94.8| 102M |
| ELECTRA-large | 95.2 | 95.3| 324M |
| ELECTRA-180g-large | 94.8 | 95.2| 324M |


# MacBERT
link：https://github.com/ymcui/MacBERT

参数：
* MacBERT-large, Chinese：24-layer, 1024-hidden, 16-heads, 324M parameters
* MacBERT-base, Chinese：12-layer, 768-hidden, 12-heads, 102M parameters

效果：任务为基于二分类的情感分类数据集ChnSentiCorp的情感分析任务，评价指标为：Accuracy
| 模型 | 开发集 | 测试集 | 参数量 |
| ---- | ---- | ---- | ---- |
| BERT-base | 94.7 (94.3) | 95.0 (94.7)| 102M |
| BERT-wwm | 95.1 (94.5) | 95.4 (95.0)| 102M |
| BERT-wwm-ext | 95.4 (94.6) | 95.3 (94.7)| 102M |
| RoBERTa-wwm-ext | 95.0 (94.6) | 95.6 (94.8)| 102M |
| ELECTRA-base | 93.8 (93.0) | 94.5 (93.5)| 102M |
| MacBERT-base | 95.2 (94.8) | 95.6 (94.9)| 102M |
| MacBERT-large | 95.7 (95.0) | 95.9 (95.1)| 324M |
| ELECTRA-large | 95.2(94.6) | 95.3(94.8)| 324M |

# Albert
link：https://github.com/brightmart/albert_zh

参数配置（与BERT对比）:![参数配置（与BERT对比）](https://github.com/brightmart/albert_zh/raw/master/resources/albert_configuration.jpg)

## tiny
效果(tiny版本)：训练和推理预测速度提升约10倍，精度基本保留，模型大小为bert的1/25；语义相似度数据集LCQMC测试集上达到85.4%，相比bert_base仅下降1.5个点。

与BERT对比：![与BERT对比](https://github.com/brightmart/albert_zh/blob/master/resources/albert_tiny_compare_s.jpg)

## base
albert_base_zh：额外训练了1.5亿个实例,即 36k steps * batch_size 4096,
参数量12M, 层数12，大小为40M

效果：参数量为bert_base的十分之一，模型大小也十分之一；在口语化描述相似性数据集LCQMC的测试集上相比bert_base下降约0.6~1个点；相比未预训练，albert_base提升14个点

## large
层数24，文件大小为64M

效果：参数量和模型大小为bert_base的六分之一；在口语化描述相似性数据集LCQMC的测试集上相比bert_base上升0.2个点

## xlarge
177k或183k(优先尝试)参数量，层数24，文件大小为230M

参数量和模型大小为bert_base的二分之一；需要一张大的显卡；完整测试对比将后续添加；batch_size不能太小，否则可能影响精度

## 模型性能对比
英文：
![模型性能](https://github.com/brightmart/albert_zh/raw/master/resources/state_of_the_art.jpg)

![模型性能](https://github.com/brightmart/albert_zh/raw/master/resources/albert_performance.jpg)

中文：
![模型性能](https://github.com/brightmart/albert_zh/raw/master/resources/crmc2018_compare_s.jpg)

# WoBERT
link：https://github.com/ZhuiyiTechnology/WoBERT

论文介绍：https://kexue.fm/archives/7758

## base
分词方式：初始化阶段，将每个词用BERT自带的Tokenizer切分为字，然后用字embedding的平均作为词embedding的初始化

训练时间：模型使用单张24G的RTX训练了100万步（大概训练了10天）

参数概览：序列长度为512，学习率为5e-6，batch_size为16，累积梯度16步，相当于batch_size=256训练了6万步左右

## plus
参数概览：maxlen=512，batch_size=256、lr=1e-5

训练时间：训练了25万步（4 * TITAN RTX，累积4步梯度，是之前的WoBERT的4倍），每1000步耗时约1580s，共训练了18天，训练acc约64%，训练loss约1.80

文本分类效果对比：
|  | IFLYTEK | TNEWS |
| ---- | ---- | ---- |
| BERT | 60.31% | 56.94%|
| WoBERT | 61.15% | 57.05%|

# WoNEZHA
link：https://github.com/ZhuiyiTechnology/WoBERT

训练细节跟WoBERT一样，区别在于位置编码方式,它使用了相对位置编码，而BERT用的是绝对位置编码，因此理论上NEZHA能处理的文本长度是无上限的


