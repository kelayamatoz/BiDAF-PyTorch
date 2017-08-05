import argparse
import json
import math
import os
import shutil
import numpy as np

from itertools import zip_longest
from functools import reduce
from pprint import pprint
from tqdm import tqdm

from bidaf.read_data import read_data, get_squad_data_filter, update_config
from bidaf.trainer import MultiGPUTrainer
from bidaf.model import BiDAF


def main(config):
    print(config.mode)
    if config.mode == 'train':
        _train(config)
    elif config.mode == 'test':
        _test(config)
    elif config.mode == 'forward':
        _forward(config)
    else:
        raise ValueError("Invalid value for 'mode': {}".format(config.mode))


def set_dirs(config):
    # create directories
    assert config.load or config.mode == 'train', "config.load must be True if not training"
    if not config.load and os.path.exists(config.out_dir):
        shutil.rmtree(config.out_dir)

    config.save_dir = os.path.join(config.out_dir, "save")
    config.log_dir = os.path.join(config.out_dir, "log")
    config.eval_dir = os.path.join(config.out_dir, "eval")
    config.answer_dir = os.path.join(config.out_dir, "answer")
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    if not os.path.exists(config.answer_dir):
        os.mkdir(config.answer_dir)
    if not os.path.exists(config.eval_dir):
        os.mkdir(config.eval_dir)


def _config_debug(config):
    if config.debug:
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.val_num_batches = 2
        config.test_num_batches = 2


def _train(config):
    data_filter = get_squad_data_filter(config)
    train_data = read_data(config, 'train', config.load, data_filter=data_filter)
    dev_data = read_data(config, 'dev', True, data_filter=data_filter)
    update_config(config, [train_data, dev_data])

    _config_debug(config)

    word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']
    word2idx_dict = train_data.shared['word2idx']
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    config.emb_mat = emb_mat

    # Construct model and trainer
    model = BiDAF(config)
    trainer = MultiGPUTrainer(config, model)

    # Begin training
    num_steps = config.num_steps or int(math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
    global_step = 0

    for batches in tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus,
                                                     num_steps=num_steps, shuffle=True, cluster=config.cluster), total=num_steps):
        # global_step = sess.run(model.global_step) + 1  # +1 because all calculations are done after step
        global_step += 1
        get_summary = global_step % config.log_period == 0
        # loss, summary, train_op = trainer.step(sess, batches, get_summary=get_summary)
        trainer.step(batches, get_summary=get_summary)
        print("Set up trainer here...")
        if get_summary:
            # graph_handler.add_summary(summary, global_step)
            print("Add summary")

        # occasional saving
        if global_step % config.save_period == 0:
            # graph_handler.save(sess, global_step=global_step)
            print("Saving")

        if not config.eval:
            continue
        # Occasional evaluation
        if global_step % config.eval_period == 0:
            num_steps = math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus))
            print("Occasional evaluation")
            # if 0 < config.val_num_batches < num_steps:
            #     num_steps = config.val_num_batches
            # e_train = evaluator.get_evaluation_from_batches(
            #     sess, tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps)
            # )
            # graph_handler.add_summaries(e_train.summaries, global_step)
            # e_dev = evaluator.get_evaluation_from_batches(
            #     sess, tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps), total=num_steps))
            # graph_handler.add_summaries(e_dev.summaries, global_step)

            # if config.dump_eval:
            #     graph_handler.dump_eval(e_dev)
            # if config.dump_answer:
            #     graph_handler.dump_answer(e_dev)

    if global_step % config.save_period != 0:
        # graph_handler.save(sess, global_step=global_step)
        print("Period saving")


def _forward(config):
    raise NotImplementedError("_forward Not implemented")


def _test(config):
    raise NotImplementedError("_test Not implemented")
