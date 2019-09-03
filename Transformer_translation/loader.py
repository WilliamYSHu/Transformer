import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
from config import Config
from preprocess import indexesFromSentence, tensorFromSentence, tensorsFromPair
import numpy as np

EOS_token = Config.EOS_token
PADDING_token = Config.PADDING_token
MAX_LENGTH = Config.MAX_LENGTH
SOS_token = Config.SOS_token


def triu_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # triangular mask with upper triangular (no diagonal) being False and others True
    return torch.from_numpy(subsequent_mask) == 0


std_triu_mask = triu_mask(MAX_LENGTH)  # standard triu mask for our setting


class TranslateDataset(Dataset):
    def __init__(self, input_lang, output_lang, pairs):
        self.input_lang = input_lang
        self.output_lang = output_lang
        self.pairs = pairs

        self.n_pairs = len(pairs)

        self.input_tensors = []
        self.output_tensors = []

        for pair in pairs:
            self.input_tensors.append(tensorFromSentence(input_lang, pair[0]))
            self.output_tensors.append(tensorFromSentence(output_lang, pair[1]))

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]


def my_collate_fn(batch):
    input_tensors, output_tensors = zip(*batch)
    batch_size = len(output_tensors)

    input_packed = pack_sequence(input_tensors, enforce_sorted=False)
    input_stacked, _ = pad_packed_sequence(input_packed, batch_first=True,
                                           padding_value=PADDING_token, total_length=MAX_LENGTH)

    output_packed = pack_sequence(output_tensors, enforce_sorted=False)
    output_stacked, _ = pad_packed_sequence(output_packed, batch_first=True,
                                            padding_value=PADDING_token, total_length=MAX_LENGTH)

    # Add SOS token to output's front
    output_stacked = torch.cat((torch.tensor(SOS_token).repeat(batch_size, 1).type_as(output_stacked),
                                output_stacked), -1)

    decoder_input = output_stacked[:, :-1]
    output_gold = output_stacked[:, 1:]

    encoder_mask = (input_stacked != PADDING_token).unsqueeze(-2)

    # shape: B * MAX_LENGTH, B. With EOS. mask: B * MAX_LENGTH * MAX_LENGTH
    return input_stacked, encoder_mask, decoder_input, decode_mask(decoder_input), output_gold


def translate_dataloader(input_lang, output_lang, pairs):
    dataset = TranslateDataset(input_lang, output_lang, pairs)
    return DataLoader(dataset, shuffle=True, batch_size=Config.batch_size, collate_fn=my_collate_fn)


def decode_mask(tgt):
    """
    get the triangular mask for the input of transformers
    :param tgt: batch_size, MAX_LEN
    :return: batch_size, MAX_LEN, MAX_LEN
    """
    # batch_size * MAX_LEN * MAX_LEN
    tgt_mask = (tgt != PADDING_token).unsqueeze(-2)
    tgt_mask = tgt_mask & std_triu_mask.type_as(tgt_mask.data)
    return tgt_mask
