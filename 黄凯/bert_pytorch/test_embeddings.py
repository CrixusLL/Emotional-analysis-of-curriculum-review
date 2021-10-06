import random
import torch

from config import config
from preProcess.tokenizer import *
from preProcess.create_preTraining_data import create_training_instances
from modeling.embedding import Embedding

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
    config.vocab_size = tokenizer.vocab_size
    config.pos_tag_size = tokenizer.pos_tag_size
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

    if config.use_cuda:
        torch.device("cuda")
    else:
        torch.device("cpu")
    embedding = Embedding(config)
    input_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(instances[0].tokens)]).type(torch.long)
    token_type_ids = torch.tensor(
        [instances[0].segment_ids]).type(torch.long)
    part_of_speech_ids = torch.tensor(
        [tokenizer.convert_pos_tags_to_ids(instances[0].pos_tag_ids)]).type(torch.long)
    embed = embedding.forward(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=None,
        inputs_embeds=None,
        part_of_speech_ids=part_of_speech_ids,
        past_key_values_length=0
    )
    print(embed)