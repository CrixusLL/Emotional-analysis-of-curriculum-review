import random

from config import config
from preProcess.tokenizer import *
from preProcess.create_preTraining_data import create_training_instances

if __name__ == "__main__":
    # tokenizer = Tokenizer(
    #     create_wordDict=True,
    #     data_directory_path="./data/miniData_debug/",
    #     filePath_word_to_idx="./data/save/word_to_idx.json",
    #     filePath_idx_to_word="./data/save/idx_to_word.json"
    # )
    tokenizer = Tokenizer(
        create_wordDict=False,
        data_directory_path=None,
        filePath_word_to_idx="./data/save/word_to_idx.json",
        filePath_idx_to_word="./data/save/idx_to_word.json"
    )
    directory_data_path = "./data/miniData_debug/"
    input_files = Tokenizer.findAllFile(directory_data_path)
    config = config()
    rng = random.Random(config.random_seed)
    instances = create_training_instances(
                input_files, tokenizer, config.max_seq_length, config.dupe_factor,
                config.short_seq_prob, config.masked_lm_prob, config.max_predictions_per_seq,
                rng)
    print("tokens:", instances[0].tokens)
    print("segment_ids:", instances[0].segment_ids)
    print("pos_tag_ids:", instances[0].pos_tag_ids)
    print("is_random_next:", instances[0].is_random_next)
    print("masked_lm_positions:", instances[0].masked_lm_positions)
    print("masked_lm_labels:", instances[0].masked_lm_labels)
    print("masked_lm_pos_tag_positions:", instances[0].masked_lm_pos_tag_positions)
    print("masked_lm_pos_tag_labels:", instances[0].masked_lm_pos_tag_labels)
    """
    输出为:
        tokens: ['[CLS]', '英国', '[MASK]', '罗素', '对', '哲学', '的', '定义', '是', '：', '胡适', '在', '《', '中国', '哲学史', '[MASK]', '》', '中', '希', '「', '凡', '研究', '人生', '切要', '[MASK]', '问题', '，', '从', '根本上', '[MASK]', '，', '要', '寻', '一个', '避免', '的', '解决', '：', '这', '种', '学问', '叫做', '哲学', '」', '。', '[SEP]', '，', '[MASK]', '50', '及', '60', '年代', '出现', '了', '以', '四', '大', '抗战', '[MASK]', '为', '代表', '的', '战斗', '文艺', '小说', '意涵', '都', '是', '以', '[MASK]', '[MASK]', '[MASK]', '背景', '，', '后来', '又', '有', '反', '共', '文学', '的', '出现', '[MASK]', '而', '60', '年代', '开始', '，', '以', '琼瑶', '为', '代表', '的', '言情', '[MASK]', '也', '开始', '行', '。', '70', '年代', '至', '逐渐', '开始', '有', '对于', '台湾', '[MASK]', '研究', '的', '新', '现代', '文学', '，', '以及', '强调', '乡土', '的', '乡土', '写实', '文学', '，', '[MASK]', '后', '也', '开始', '了', '[SEP]']
        segment_ids: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        pos_tag_ids: ['u', 'u', '[MASK]', 'u', 'u', 'n', 'u', 'n', 'u', 'u', 'u', 'u', 'u', 'u', 'n', '[MASK]', 'u', 'u', 'n', 'u', 'u', 'u', 'n', 'u', '[MASK]', 'n', 'u', 'u', 'n', '[MASK]', 'u', 'u', 'u', 'u', 'u', 'u', 'n', 'u', 'u', 'u', 'n', 'u', 'n', 'u', 'u', 'u', 'u', '[MASK]', 'u', 'u', 'u', 'n', 'u', 'u', 'u', 'u', 'a', 'n', '[MASK]', 'u', 'n', 'u', 'n', 'n', 'n', '[MASK]', 'u', 'u', 'u', '[MASK]', '[MASK]', '[MASK]', 'n', 'u', 'u', 'u', 'u', 'u', 'u', 'n', 'u', 'n', '[MASK]', 'u', 'u', 'n', 'u', 'u', 'u', 'u', 'u', 'n', 'u', 'n', '[MASK]', 'u', 'u', 'u', 'u', 'u', 'n', 'n', 'u', 'u', 'u', 'u', 'u', '[MASK]', 'n', 'u', 'a', 'a', 'n', 'u', 'u', 'u', 'n', 'u', 'n', 'n', 'n', 'u', '[MASK]', 'u', 'u', 'u', 'u', 'u']
        is_random_next: True
        masked_lm_positions: [2, 15, 18, 24, 29, 34, 47, 57, 58, 65, 69, 70, 71, 82, 94, 101, 107, 122, 125]
        masked_lm_labels: ['哲学家', '大纲', '称', '的', '着想', '根本', '在', '抗战', '小说', '，', '抗战', '时期', '为', '，', '小说', '起', '社会', '1990年', '开始']
        masked_lm_pos_tag_positions: [2, 15, 18, 24, 29, 34, 47, 57, 58, 65, 69, 70, 71, 82, 94, 101, 107, 122, 125]
        masked_lm_pos_tag_labels: ['n', 'n', 'u', 'u', 'u', 'a', 'u', 'n', 'n', 'u', 'n', 'n', 'u', 'u', 'n', 'u', 'n', 'u', 'u']
    """
    print("done")