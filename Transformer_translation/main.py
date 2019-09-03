import torch
import torch.nn as nn
from preprocess import prepareData
from loader import translate_dataloader
from transformer import Transformer
from config import Config
from train import train_one_epoch
from torch.optim import Adam
from loss import LabelSmoothingLoss
from optim import NoamOpt
from evaluate import evaluate_dataset, evaluateRandomly

# data
input_lang, output_lang, pairs_train, pairs_dev = prepareData('eng', 'fra', 'eng-fra.txt', True)

train_loader = translate_dataloader(input_lang, output_lang, pairs_train)
dev_loader = translate_dataloader(input_lang, output_lang, pairs_dev)

# model
model = Transformer(input_lang.n_words, output_lang.n_words, Config.n_enc_layer, Config.n_dec_layer,
                          Config.hidden_size, Config.num_head, Config.head_size, Config.feedforward_size,
                          Config.dropout, Config.attn_dropout, Config.layer_norm_eps)
model.cuda()

# load model if pretrained
if Config.pretrained:
    model.load_state_dict(torch.load(Config.model_load))

# optimizer
optimizer = NoamOpt(Config.hidden_size, Config.factor, Config.warmup,
                    Adam(model.parameters(), lr=Config.lr, betas=(0.9, 0.98), eps=1e-9))

# criterion
criterion = LabelSmoothingLoss(0.1, tgt_vocab_size=output_lang.n_words, ignore_index=Config.PADDING_token).cuda()
#criterion = nn.NLLLoss()

# training
best_bleu = -1
for i in range(Config.n_epoch):
    train_one_epoch(train_loader, model, optimizer, criterion, print_every=50)
    if i % 5 == 0:
        evaluateRandomly(pairs_dev, input_lang, output_lang, model, n=3)
    acc, bleu = evaluate_dataset(dev_loader, model, output_lang)
    print("accuracy: {}  bleu score: {}".format(acc, bleu))

    # save model if best
    if best_bleu < bleu:
        best_bleu = bleu
        torch.save(model.state_dict(), "ckpts/transformer_{}.pt".format(i))
