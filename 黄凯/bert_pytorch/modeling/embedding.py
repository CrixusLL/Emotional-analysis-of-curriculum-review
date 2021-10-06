import torch
import torch.nn as nn
from packaging import version

class Embedding(nn.Module):
    def __init__(self, config):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)  # token embedding
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)  # position embedding
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)  # token type embedding
        self.part_of_speech_embeddings = nn.Embedding(config.pos_tag_size, config.hidden_size) # part of speech embedding

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "token_type_ids",
                torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
                persistent=False,
            )

    # def forward(self, x, seg, pos_tag_ids=None):
    #     seq_len = x.size(1)
    #     pos = torch.arange(seq_len, dtype=torch.long)
    #     pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
    #     embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg) + self.partOfSpeech_embed(pos_tag_ids)
    #     embedding = self.norm(embedding)
    #     embedding = self.dropout(embedding)
    #     return self.dropout(self.norm(embedding))

    def forward(
        self, 
        input_ids=None, 
        token_type_ids=None, 
        position_ids=None, 
        inputs_embeds=None, 
        part_of_speech_ids=None,
        past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        if part_of_speech_ids is None:
            pass
        else:
            part_of_speech_embeddings = self.part_of_speech_embeddings(part_of_speech_ids)
            embeddings += part_of_speech_embeddings
            
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
