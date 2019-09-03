import unicodedata
import re
from config import Config
import torch
import numpy as np

SOS_token = Config.SOS_token
EOS_token = Config.EOS_token


# include word counts of a language
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, fn, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(fn, encoding='utf-8'). \
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


MAX_LENGTH = Config.MAX_LENGTH

eng_prefixes = Config.eng_prefixes


# SOS and EOS
def filterPair(p):
    if Config.LIMIT_PREFIX:
        return len(p[0].split(' ')) < MAX_LENGTH-1 \
               and len(p[1].split(' ')) < MAX_LENGTH-1 \
               and p[1].startswith(eng_prefixes)
    else:
        return len(p[0].split(' ')) < MAX_LENGTH-1 \
               and len(p[1].split(' ')) < MAX_LENGTH-1


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, fn, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, fn, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    n_pair = len(pairs)
    indices = list(torch.randperm(n_pair))
    train_indices = indices[:int((1-Config.dev_split)*n_pair)]
    dev_indices = indices[int((1-Config.dev_split)*n_pair):]
    pairs = np.array(pairs)

    return input_lang, output_lang, list(pairs[train_indices]), list(pairs[dev_indices])


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long)


def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return input_tensor, target_tensor
