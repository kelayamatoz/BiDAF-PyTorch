import torch
import torch.optim as O
import os.path
from model import BIDAF
from torchtext import data, datasets


def train_batch(config, model, opt, batch):
    model.train(); opt.zero_grad()
    n_correct, n_total = 0, 0
    print(batch)
    ls, s, le, e = model(batch)


def train_epoch(config, model, opt, iter):
    iter.init_epoch()
    n_correct, n_total = 0, 0
    for batch_idx, batch in enumerate(iter):
        iterations += 1
        n_correct, n_total = train_batch(config, model, opt, batch)


def train(config):
    print("building context vocab")

    train_iter, dev_iter, test_iter = datasets.SQUAD.iters(config,
            batch_size=config.batch_size, device=config.gpu)


    model = BIDAF(train_iter.config)
    model.word_embed.weight.data = context.vocab.vectors

    if not config.cpu:
        model = model.cuda()

    opt = O.Adam(model.parameters())

    print("training")
    for epoch in range(config.epochs):
        train_epoch(config, model, opt, train_iter)
