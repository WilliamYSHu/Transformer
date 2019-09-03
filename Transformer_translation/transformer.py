import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from config import Config

half_precision_flag = Config.half_precision
MAX_LENGTH = Config.MAX_LENGTH
PADDING_token = Config.PADDING_token


class EmbeddingLayer(nn.Module):

    def __init__(self, input_size, hidden_size, max_length, padding_idx, dropout):
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=hidden_size,
            padding_idx=padding_idx)
        self.position_embeddings = torch.nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=hidden_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids):
        """
        :param input_ids: batch_size, MAX_LENGTH
        :return: batch_size, MAX_LENGTH, hidden_dim
        """

        # max_length
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        # batch size, max_length
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # batch size, sequence length, embedding dimension
        words_embeddings = self.word_embeddings(input_ids)

        # batch size, sequence length, embedding dimension
        positions_embeddings = self.position_embeddings(position_ids)

        # batch size, sequence length, embedding dimension
        embeddings = words_embeddings + positions_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class DotProductAttention(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)  # dropout in attention

    def forward(self, query, key, value, mask):
        """
        :param query, key, value:  batch_size, num_head, max_len, head_size
        :param mask: batch_size, max_len, max_len (for position in dim -2, the mask in dim -1)
        :return: batch_size, num_head, max_len, head_size
        """
        # batch_size, num_head, max_len (output), max_len (input)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])

        # mask values
        mask_val = -1e4 if half_precision_flag else -1e9
        mask = mask.unsqueeze(1)  # batch_size, 1, max_len, max_len
        attention_scores = attention_scores.masked_fill(mask == 0, mask_val)

        attention_probs = F.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        # output
        return torch.matmul(attention_probs, value)


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_head, head_size, hidden_size, attn_dropout):
        super().__init__()
        self.head_size = head_size
        self.num_head = num_head
        self.dot_attention = DotProductAttention(attn_dropout)
        self.query = nn.Linear(hidden_size, num_head * head_size)
        self.key = nn.Linear(hidden_size, num_head * head_size)
        self.value = nn.Linear(hidden_size, num_head * head_size)
        self.out = torch.nn.Linear(num_head * head_size, hidden_size)

    def forward(self, queries, keys, values, mask):
        """
        :param queries, keys, values: batch * max_len * hidden_size
        :param mask: batch * max_len * max_len
        :return: batch * max_len * hidden_size
        """
        batch_size = queries.shape[0]
        queries = self.query(queries).view(batch_size, self.num_head, -1, self.head_size)
        keys = self.key(keys).view(batch_size, self.num_head, -1, self.head_size)
        values = self.value(values).view(batch_size, self.num_head, -1, self.head_size)

        context = self.dot_attention(queries, keys, values, mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_head * self.head_size)
        return self.out(context)


class FeedForward(nn.Module):

    def __init__(self, hidden_size, feedforward_size, dropout):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, feedforward_size)
        self.w2 = nn.Linear(feedforward_size, hidden_size)

        # Bert's special activation function
        def activation_fn(x):
            return x * .5 * (1. + torch.erf(x / math.sqrt(2.)))

        self.act_fun = activation_fn

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_state):
        """
        :param input_state: batch_size * max_len * hidden_size
        :return: same as the input
        """
        return self.w2(self.dropout(self.act_fun(self.w1(input_state))))


class SublayerConnection(nn.Module):

    def __init__(self, hidden_size, dropout, eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_state, sublayer):
        """
        :param input_state: batch_size * max_len * hidden_size
        :param sublayer: a torch.nn.Module that returns output with the same size
        :return: same as the input
        """
        return input_state + self.dropout(sublayer(self.layer_norm(input_state)))


class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, num_head, head_size, feedforward_size, dropout, attn_dropout, layer_norm_eps):
        super().__init__()
        self.self_attention = MultiHeadedAttention(num_head, head_size, hidden_size, attn_dropout)
        self.feedforward = FeedForward(hidden_size, feedforward_size, dropout)
        self.sub_layer = nn.ModuleList([copy.deepcopy(SublayerConnection(hidden_size, dropout, layer_norm_eps))
                                        for _ in range(2)])

    def forward(self, hidden_state, mask):
        """
        :param hidden_state: batch_size * max_len * hidden_size
        :param mask: batch_size * max_len * max_len
        :return: same as input
        """
        hidden_state = self.sub_layer[0](hidden_state, lambda x: self.self_attention(x, x, x, mask))
        return self.sub_layer[1](hidden_state, self.feedforward)


class Encoder(nn.Module):

    def __init__(self, n_layer, hidden_size, num_head, head_size, feedforward_size,
                 dropout, attn_dropout, layer_norm_eps):
        super().__init__()
        attn_block = EncoderLayer(hidden_size, num_head, head_size, feedforward_size,
                                  dropout, attn_dropout, layer_norm_eps)
        self.attn_blocks = nn.ModuleList([copy.deepcopy(attn_block) for _ in range(n_layer)])
        self.last_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)

    def forward(self, hidden_state, mask):
        """
        :param hidden_state: batch_size * max_len * hidden_size
        :param mask: batch_size * max_len * max_len
        :return: same as input
        """
        for attn_block in self.attn_blocks:
            hidden_state = attn_block(hidden_state, mask)
        return self.last_layer_norm(hidden_state)


class DecoderLayer(nn.Module):

    def __init__(self, hidden_size, num_head, head_size, feedforward_size, dropout, attn_dropout, layer_norm_eps):
        super().__init__()
        self.cross_attention = MultiHeadedAttention(num_head, head_size, hidden_size, attn_dropout)
        self.feedforward = FeedForward(hidden_size, feedforward_size, dropout)
        self.sub_layer = nn.ModuleList([copy.deepcopy(SublayerConnection(hidden_size, dropout, layer_norm_eps))
                                        for _ in range(3)])

    def forward(self, hidden_state, encode_state, src_mask, tgt_mask):
        """
        :param hidden_state: batch_size * max_len * hidden_size
        :param encode_state: batch_size * max_len * hidden_size
        :param src_mask, tgt_mask: batch_size * max_len * max_len
        :return: same as hidden_state
        """

        hidden_state = self.sub_layer[0](hidden_state, lambda x: self.cross_attention(x, x, x, tgt_mask))

        hidden_state = self.sub_layer[1](hidden_state, lambda x: self.cross_attention(x, encode_state, encode_state,
                                                                                      src_mask))

        return self.sub_layer[2](hidden_state, self.feedforward)


class Decoder(nn.Module):

    def __init__(self, n_layer, hidden_size, num_head, head_size, feedforward_size,
                 dropout, attn_dropout, layer_norm_eps):
        super().__init__()
        attn_block = DecoderLayer(hidden_size, num_head, head_size, feedforward_size,
                                  dropout, attn_dropout, layer_norm_eps)
        self.attn_blocks = nn.ModuleList([copy.deepcopy(attn_block) for _ in range(n_layer)])
        self.last_layer_norm = nn.LayerNorm(hidden_size, layer_norm_eps)

    def forward(self, hidden_state, encoder_state, src_mask, tgt_mask):
        """
        :param hidden_state, encoder_state: batch_size * max_len * hidden_size
        :param src_mask, tgt_mask: batch_size * max_len * max_len
        :return: same as input
        """
        for attn_block in self.attn_blocks:
            hidden_state = attn_block(hidden_state, encoder_state, src_mask, tgt_mask)
        return self.last_layer_norm(hidden_state)


class Transformer(nn.Module):

    def __init__(self, input_size, output_size, n_enc_layer, n_dec_layer, hidden_size,
                 num_head, head_size, feedforward_size, dropout, attn_dropout, layer_norm_eps):
        super().__init__()

        # embeddings
        self.input_embed = EmbeddingLayer(input_size, hidden_size, MAX_LENGTH, PADDING_token, dropout)
        self.output_embed = EmbeddingLayer(output_size, hidden_size, MAX_LENGTH, PADDING_token, dropout)

        # encoders and decoders
        self.encoder = Encoder(n_enc_layer, hidden_size, num_head, head_size, feedforward_size,
                               dropout, attn_dropout, layer_norm_eps)
        self.decoder = Decoder(n_dec_layer, hidden_size, num_head, head_size, feedforward_size,
                               dropout, attn_dropout, layer_norm_eps)

        # Final layer
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_input, encoder_mask, decoder_input, decoder_mask):
        """
        :param encoder_input: batch_size , MAX_LENGTH
        :param encoder_mask: batch_size , MAX_LENGTH, MAX_LENGTH
        :param decoder_input: batch_size , MAX_LENGTH
        :param decoder_mask: batch_size , MAX_LENGTH, MAX_LENGTH
        :return: batch_size * MAX_LENGTH , output_size
        """
        encode_tensor = self.encoder_forward(encoder_input, encoder_mask)
        return self.decoder_forward(decoder_input, decoder_mask, encode_tensor, encoder_mask)

    def encoder_forward(self, encoder_input, encoder_mask):
        # batch_size * max_len * hidden_size
        encode_tensor = self.input_embed(encoder_input)
        return self.encoder(encode_tensor, encoder_mask)

    def decoder_forward(self, decoder_input, decoder_mask, encoder_tensor, encoder_mask):
        # batch_size * max_len * hidden_size
        decode_tensor = self.output_embed(decoder_input)
        decode_tensor = self.decoder(decode_tensor, encoder_tensor, encoder_mask, decoder_mask)

        # batch_size * max_len * output_size
        decode_tensor = self.out(decode_tensor)
        return F.log_softmax(decode_tensor, dim=-1)
