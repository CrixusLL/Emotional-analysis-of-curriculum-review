import collections
import random
import json

from .tokenizer import Tokenizer, HanLP

class TrainingInstance(object):
    """A single training instance (sentence pair)."""
    def __init__(
        self, 
        tokens, 
        segment_ids,
        pos_tag_ids,
        masked_lm_positions, 
        masked_lm_labels,
        masked_lm_pos_tag_positions,
        masked_lm_pos_tag_labels,
        is_random_next
    ):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.pos_tag_ids = pos_tag_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.masked_lm_pos_tag_positions = masked_lm_pos_tag_positions
        self.masked_lm_pos_tag_labels = masked_lm_pos_tag_labels

def loadData_segmentByHanLP(paths):
    documents = []
    pos_tag_documents = []
    for path in paths:
        f = open(path)
        s = f.readlines()
        for line in s:
            document = []
            pos_tag_document = []
            for sentence in json.loads(line)["text"].split('\n'):
                if sentence:
                    document.append(HanLP(sentence)["tok/fine"])
                    pos_tag_document.append([Tokenizer.convert_pos_tag(_pos_tag) for _pos_tag in HanLP(sentence)['pos/ctb']])
            documents.append(document)
            pos_tag_documents.append(pos_tag_document)
    return documents, pos_tag_documents

def create_training_instances(
        input_files, tokenizer, max_seq_length, 
        dupe_factor, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, rng
    ):
    """Create `TrainingInstance`s from raw text."""

    all_documents = [[]]
    all_pos_tag_documents = [[]]
    all_documents, all_pos_tag_documents = loadData_segmentByHanLP(input_files)
    # rng.shuffle(all_documents)
    tokenAndpos_tag_documents = list(zip(all_documents, all_pos_tag_documents))
    rng.shuffle(tokenAndpos_tag_documents)
    all_documents[:], all_pos_tag_documents[:] = zip(*tokenAndpos_tag_documents)

    vocab_words = list(tokenizer.word_to_idx.keys())
    type_pos_tags = list(tokenizer.pos_to_idx.keys())

    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, all_pos_tag_documents,
                    document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, type_pos_tags,
                    rng))

    rng.shuffle(instances)
    return instances

def create_instances_from_document(
        all_documents, all_pos_tag_documents,
        document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, type_pos_tags,
        rng
    ):
    """Creates `TrainingInstance`s for a single document."""

    document = all_documents[document_index]
    pos_tag_document = all_pos_tag_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_chunk_pos_tags = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        segment_pos_tags = pos_tag_document[i]
        current_chunk_pos_tags.append(segment_pos_tags)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                pos_tags_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                    pos_tags_a.extend(current_chunk_pos_tags[j])

                tokens_b = []
                pos_tags_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break
                    
                    random_document = all_documents[random_document_index]
                    random_pos_tag_document = all_pos_tag_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        pos_tags_b.extend(random_pos_tag_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                        pos_tags_b.extend(current_chunk_pos_tags[j])

                # 掐头 / 去尾
                truncate_seq_pair(
                    tokens_a, tokens_b, 
                    pos_tags_a, pos_tags_b,
                    max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1
                assert len(pos_tags_a) >= 1
                assert len(pos_tags_b) >= 1

                tokens = []
                pos_tags = []
                segment_ids = []
                tokens.append("[CLS]")
                pos_tags.append("u")
                segment_ids.append(0)
                for token, pos_tag in zip(tokens_a, pos_tags_a):
                    tokens.append(token)
                    pos_tags.append(pos_tag)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                pos_tags.append("u")
                segment_ids.append(0)

                for token, pos_tag in zip(tokens_b, pos_tags_b):
                    tokens.append(token)
                    pos_tags.append(pos_tag)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                pos_tags.append("u")
                segment_ids.append(1)

                (tokens, tokens_pos_tags,
                masked_lm_positions,
                masked_lm_labels,
                masked_lm_pos_tag_positions,
                masked_lm_pos_tag_labels) = create_masked_lm_predictions(
                                        tokens, pos_tags,
                                        masked_lm_prob, max_predictions_per_seq, 
                                        vocab_words, type_pos_tags,
                                        rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    pos_tag_ids=tokens_pos_tags,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    masked_lm_pos_tag_positions=masked_lm_pos_tag_positions,
                    masked_lm_pos_tag_labels=masked_lm_pos_tag_labels
                )
                instances.append(instance)
            current_chunk = []
            current_chunk_pos_tags = []
            current_length = 0
        i += 1
    
    return instances

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

MaskedLm_pos_tag_Instance = collections.namedtuple("MaskedLm_pos_tag_Instance",
                                          ["index", "label"])

def create_masked_lm_predictions(
        tokens, pos_tags,
        masked_lm_prob, max_predictions_per_seq, 
        vocab_words, type_pos_tags,
        rng
    ):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)
    output_tokens_pos_tags = list(pos_tags)

    num_to_predict = min(max_predictions_per_seq,
                        max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    masked_lms_pos_tag = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break

        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            masked_token_pos_tag = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
                masked_token_pos_tag = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                    masked_token_pos_tag = pos_tags[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]
                    masked_token_pos_tag = type_pos_tags[rng.randint(0, len(type_pos_tags) - 1)]

            output_tokens[index] = masked_token
            output_tokens_pos_tags[index] = masked_token_pos_tag

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            masked_lms_pos_tag.append(MaskedLm_pos_tag_Instance(index=index, label=pos_tags[index]))
    assert len(masked_lms) <= num_to_predict
    assert len(masked_lms_pos_tag) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lms_pos_tag = sorted(masked_lms_pos_tag, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    masked_lm_pos_tag_positions = []
    masked_lm_pos_tag_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    for p in masked_lms_pos_tag:
        masked_lm_pos_tag_positions.append(p.index)
        masked_lm_pos_tag_labels.append(p.label)

    return (
        output_tokens, output_tokens_pos_tags,
        masked_lm_positions, masked_lm_labels, 
        masked_lm_pos_tag_positions, masked_lm_pos_tag_labels)

def truncate_seq_pair(
    tokens_a, tokens_b, 
    pos_tags_a, pos_tags_b,
    max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        trunc_tokens_pos_tags = pos_tags_a if len(pos_tags_a) > len(pos_tags_b) else pos_tags_b
        assert len(trunc_tokens) >= 1
        assert len(trunc_tokens_pos_tags) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
            del trunc_tokens_pos_tags[0]
        else:
            trunc_tokens.pop()
            trunc_tokens_pos_tags.pop()
