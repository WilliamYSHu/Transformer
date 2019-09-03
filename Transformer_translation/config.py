class Config:
    # preprocess and data
    PADDING_token = 0
    SOS_token = 1
    EOS_token = 2
    MAX_LENGTH = 10  # include EOS token
    eng_prefixes = (
        "i am ", "i m ",
        "he is", "he s ",
        "she is", "she s ",
        "you are", "you re ",
        "we are", "we re ",
        "they are", "they re "
    )
    LIMIT_PREFIX = False
    dev_split = 0.2

    # training
    half_precision = False
    batch_size = 384
    teacher_forcing_ratio = 0.5
    n_epoch = 50

    # network
    n_enc_layer = 1
    n_dec_layer = 1
    hidden_size = 64
    num_head = 1
    head_size = hidden_size//num_head
    feedforward_size = 2 * hidden_size
    dropout = 0.1
    attn_dropout = 0.1
    layer_norm_eps = 1e-9

    # optimizer
    lr = 1e-4
    factor = 1
    warmup= 2000

    # save model
    pretrained = False
    model_load = 'ckpts/transformer_21.pt'


