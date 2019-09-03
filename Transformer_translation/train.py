import torch
import torch.nn as nn
from config import Config
import random
import time
import math

MAX_LENGTH = Config.MAX_LENGTH
PADDING_token = Config.PADDING_token
SOS_token = Config.SOS_token


def train_one_batch(encoder_input, encoder_mask, decoder_input, decoder_mask, output_gold,
                    transformer, opt, criterion):
    transformer.train()
    opt.optimizer.zero_grad()

    encoder_input = encoder_input.type(torch.long).cuda()
    encoder_mask = encoder_mask.type(torch.ByteTensor).cuda()
    decoder_input = decoder_input.type(torch.long).cuda()
    decoder_mask = decoder_mask.type(torch.ByteTensor).cuda()
    output_gold = output_gold.type(torch.long).cuda()

    model_output = transformer(encoder_input, encoder_mask, decoder_input, decoder_mask)

    vocab_size = model_output.shape[-1]

    # compute loss
    loss = criterion(model_output.view(-1, vocab_size), output_gold.view(-1))

    loss.backward()

    opt.step()
    opt.optimizer.step()

    return loss.item()


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train_one_epoch(data_loader, transformer, opt, criterion, print_every=1000):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    n_batch = len(data_loader)

    for iterate, (input_stacked, encoder_mask, decoder_input, decoder_mask, output_gold) in enumerate(data_loader):

        loss = train_one_batch(input_stacked, encoder_mask, decoder_input, decoder_mask, output_gold,
                               transformer, opt, criterion)

        print_loss_total += loss
        if iterate % print_every == 0 and iterate != 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iterate / n_batch),
                                         iterate, iterate / n_batch * 100, print_loss_avg))


