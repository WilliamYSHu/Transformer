import torch
from config import Config
import random
from nltk.translate.bleu_score import sentence_bleu
from preprocess import tensorFromSentence
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from loader import decode_mask

MAX_LENGTH = Config.MAX_LENGTH
PADDING_token = Config.PADDING_token
SOS_token = Config.SOS_token
EOS_token = Config.EOS_token


def evaluate_one_batch(encoder_input, encoder_mask, transformer):
    """
    :param stacked_input: batch_size, MAX_LEN
    :param input_mask:  batch_size, MAX_LEN, MAX_LEN
    :param transformer: Transformer object. also contains encoder and decoder
    :return:
    """
    transformer.eval()

    encoder_input = encoder_input.type(torch.long).cuda()
    encoder_mask = encoder_mask.type(torch.ByteTensor).cuda()

    decoder_input = torch.zeros_like(encoder_input).fill_(PADDING_token)
    decoder_input[:, 0].fill_(SOS_token)
    decoder_mask = decode_mask(decoder_input)

    with torch.no_grad():

        # Greedy inference
        memory = transformer.encoder_forward(encoder_input, encoder_mask)

        for i in range(1, MAX_LENGTH):
            # batch_size * max_len * output_size
            out = transformer.decoder_forward(decoder_input, decoder_mask, memory, encoder_mask)
            _, next_word = torch.max(out[:, i-1], dim=-1)  # batch_size * max_len
            decoder_input[:, i] = next_word
            decoder_mask = decode_mask(decoder_input)

        # last token
        out = transformer.decoder_forward(decoder_input, decoder_mask, memory, encoder_mask)
        _, next_word = torch.max(out[:, -1], dim=-1)  # batch_size * max_len
        decoder_input[:, :-1] = decoder_input[:, 1:].clone()
        decoder_input[:, -1] = next_word

    return decoder_input.cpu().numpy()  # not contains SOS


def evaluate_single_pair(input_sent, input_lang, output_lang, transformer):
    input_sent = tensorFromSentence(input_lang, input_sent)
    input_packed = pack_sequence([input_sent], enforce_sorted=False).cuda()
    input_stacked, _ = pad_packed_sequence(input_packed, batch_first=True,
                                           padding_value=PADDING_token, total_length=MAX_LENGTH)
    encoder_mask = (input_stacked != PADDING_token).unsqueeze(-2)

    results = evaluate_one_batch(input_stacked, encoder_mask, transformer)
    return stacked_to_lst_of_sentence(results, output_lang)[0]


def evaluateRandomly(pairs, input_lang, output_lang, transformer, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate_single_pair(pair[0], input_lang, output_lang, transformer)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# accuracy and bleu score scorer
def evaluate_dataset(data_loader, transformer, output_lang):
    total_sentence = 0
    total_bleu_score = 0
    total_token = 0
    matched_token = 0

    for input_stacked, encoder_mask, decoder_input, decoder_mask, output_gold in data_loader:
        results = evaluate_one_batch(input_stacked, encoder_mask, transformer)
        n_sentence = results.shape[0]
        results_list = stacked_to_lst_of_sentence(results, output_lang)
        answer_list = stacked_to_lst_of_sentence(output_gold.cpu().numpy(), output_lang)
        for i in range(n_sentence):
            total_sentence += 1
            infer_result = results_list[i]
            answer = answer_list[i]
            total_token += len(answer)
            this_score = sentence_bleu([infer_result], answer)
            total_bleu_score += this_score
            matched_token += count_matched_token(infer_result, answer)
    return matched_token / total_token, 100 * total_bleu_score / total_sentence


# tensor to word result
def stacked_to_lst_of_sentence(stacked_input, output_lang):
    results = []
    batch_size, max_len = stacked_input.shape
    for i in range(batch_size):
        this_sentence = []
        for j in range(max_len):        # ignore first SOS
            if stacked_input[i, j] == PADDING_token:
                break
            elif stacked_input[i, j] == EOS_token:
                this_sentence.append('<EOS>')
                break
            else:
                this_sentence.append(output_lang.index2word[stacked_input[i, j]])
        results.append(this_sentence)
    return results


# get accuracy
def count_matched_token(str_infer, str_gold):
    len1 = len(str_infer)
    len2 = len(str_gold)
    minlen = min(len1, len2)

    match_count = 0
    for i in range(minlen):
        if str_infer[i] == str_gold[i]:
            match_count += 1
    return match_count
