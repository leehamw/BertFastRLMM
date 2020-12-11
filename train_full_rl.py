""" full training (train rnn-ext + abs + RL) """
import argparse
import json
import pickle as pkl
import os
from os.path import join, exists
from itertools import cycle

from toolz.sandbox.core import unzip
from cytoolz import identity

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from pytorch_pretrained_bert import BertTokenizer

from data.data import CnnDmDataset
from data.batcher import tokenize

from model.rl import ActorCritic, BertActorCritic
from model.extract import PtrExtractSumm, BertPtrExtractSumm

from training import BasicTrainer
from rl import get_grad_fn
from rl import A2CPipeline
from decoding import load_best_ckpt
from decoding import Abstractor, ArticleBatcher, BertArticleBatcher
from metric import compute_rouge_l, compute_rouge_n


MAX_ABS_LEN = 30

try:
    DATA_DIR = '/home/wanglihan/CNN/finished_files/'
except KeyError:
    print('please use environment variable to specify data directories')


class RLDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        return art_sents, abs_sents

def load_ext_net(ext_dir, aux_device, args):
    ext_meta = json.load(open(join(ext_dir, 'meta.json')))
    assert ext_meta['net'] == 'ml_rnn_extractor'
    ext_ckpt = load_best_ckpt(ext_dir)
    ext_args = ext_meta['net_args']
    vocab = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))

    use_bert = 'bert_type' in ext_args.keys()
    fix_bert = getattr(ext_args, 'fix_bert', args.fix_bert)
    ext_args['fix_bert'] = fix_bert
    ext_args['aux_device'] = aux_device
    if use_bert:
        print('Use Bert based Extractor ...')
        ext = BertPtrExtractSumm(**ext_args)
        bert_type = ext_args['bert_type']
        tokenizer_cache = ext_args['tokenizer_cache']
        bert_config = (bert_type, tokenizer_cache)
    else:
        ext = PtrExtractSumm(**ext_args)
        bert_config = None

    ext.load_state_dict(ext_ckpt)
    return ext, vocab, use_bert, bert_config


def configure_net(abs_dir, ext_dir, cuda, aux_device, bert_max_len, args):
    """ load pretrained sub-modules and build the actor-critic network"""
    # load pretrained abstractor model
    if abs_dir is not None:
        abstractor = Abstractor(abs_dir, MAX_ABS_LEN, cuda)
    else:
        abstractor = identity

    # load ML trained extractor net and build RL agent
    extractor, agent_vocab, use_bert, bert_config = load_ext_net(ext_dir, aux_device, args)

    if use_bert:
        bert_type, tokenizer_cache = bert_config
        bert_tokenizer = BertTokenizer.from_pretrained(bert_type, cache_dir=tokenizer_cache)
        agent = BertActorCritic(extractor._sent_enc,
                                extractor._art_enc,
                                extractor._extractor,
                                BertArticleBatcher(bert_tokenizer, bert_max_len,
                                                   args.bert_max_sent, cuda),
                                cuda=cuda,
                                aux_device=aux_device)
        
    else:
        agent = ActorCritic(extractor._sent_enc,
                            extractor._art_enc,
                            extractor._extractor,
                            ArticleBatcher(agent_vocab, cuda))
        if cuda:
            agent = agent.cuda()

    net_args = {}
    net_args['abstractor'] = (None if abs_dir is None
                              else json.load(open(join(abs_dir, 'meta.json'))))
    net_args['extractor'] = json.load(open(join(ext_dir, 'meta.json')))

    return agent, agent_vocab, abstractor, net_args


def configure_training(opt, lr, clip_grad, lr_decay, batch_size,
                       gamma, reward, stop_coeff, stop_reward):
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay
    train_params['gamma']          = gamma
    train_params['reward']         = reward
    train_params['stop_coeff']     = stop_coeff
    train_params['stop_reward']    = stop_reward

    return train_params

def build_batchers(batch_size):
    def coll(batch):
        art_batch, abs_batch = unzip(batch)
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        return art_sents, abs_sents
    loader = DataLoader(
        RLDataset('train'), batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=coll
    )
    val_loader = DataLoader(
        RLDataset('val'), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll
    )
    return cycle(loader), val_loader


def train(args):
    if not exists(args.path):
        os.makedirs(args.path)

    if args.cuda:
        aux_device = torch.device('cuda', args.aux_cuda)
    else:
        aux_device = None

    # make net
    agent, agent_vocab, abstractor, net_args = configure_net(
        args.abs_dir, args.ext_dir, args.cuda, aux_device, args.bert_max_sent_len, args)

    # configure training setting
    assert args.stop > 0
    train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch,
        args.gamma, args.reward, args.stop, 'rouge-1'
    )
    train_batcher, val_batcher = build_batchers(args.batch)
    # TODO different reward
    reward_fn = compute_rouge_l
    stop_reward_fn = compute_rouge_n(n=1)

    # save abstractor binary
    if args.abs_dir is not None:
        abs_ckpt = {}
        abs_ckpt['state_dict'] = load_best_ckpt(args.abs_dir)
        abs_vocab = pkl.load(open(join(args.abs_dir, 'vocab.pkl'), 'rb'))
        abs_dir = join(args.path, 'abstractor')
        os.makedirs(join(abs_dir, 'ckpt'))
        with open(join(abs_dir, 'meta.json'), 'w') as f:
            json.dump(net_args['abstractor'], f, indent=4)
        torch.save(abs_ckpt, join(abs_dir, 'ckpt/ckpt-0-0'))
        with open(join(abs_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(abs_vocab, f)
    # save configuration
    meta = {}
    meta['net']           = 'rnn-ext_abs_rl'
    meta['net_args']      = net_args
    meta['train_params']  = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    with open(join(args.path, 'agent_vocab.pkl'), 'wb') as f:
        pkl.dump(agent_vocab, f)

    # prepare trainer
    grad_fn = get_grad_fn(agent, args.clip)
    optimizer = optim.Adam(agent.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    pipeline = A2CPipeline(meta['net'], agent, abstractor,
                           train_batcher, val_batcher,
                           optimizer, grad_fn,
                           reward_fn, args.gamma,
                           stop_reward_fn, args.stop)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, args.cumul_batch, scheduler,
                           val_mode='score')

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='program to demo a Seq2Seq model'
    )
    parser.add_argument('--path',  help='root of the model',default='/home/wanglihan/save/model')


    # model options
    parser.add_argument('--abs_dir', action='store',
                        help='pretrained summarizer model root path',default='/home/wanglihan/abstractor/model')
    parser.add_argument('--ext_dir', action='store',
                        help='root of the extractor model',default='/home/wanglihan/extractor/model')
    parser.add_argument('--ckpt', type=int, action='store', default=None,
                        help='ckeckpoint used decode')

    # bert model options
    parser.add_argument('--bert_max_sent_len', type=int, action='store', default=512,
                        help='the max sentence length used in bert encoding')
    parser.add_argument('--bert_max_sent', type=int, action='store', default=60,
                        help='maximun sentences in an article article')
    parser.add_argument('--fix_bert', action='store_true',
                        help='fix bert embedding w/o training')

    # training options
    parser.add_argument('--reward', action='store', default='rouge-l',
                        help='reward function for RL')
    parser.add_argument('--lr', type=float, action='store', default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--gamma', type=float, action='store', default=0.95,
                        help='discount factor of RL')
    parser.add_argument('--stop', type=float, action='store', default=1.0,
                        help='stop coefficient for rouge-1')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--cumul_batch', type=int, action='store', default=1,
                        help='the training batch size')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=1000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=3,
                        help='patience for early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--aux_cuda', action='store', type=int, default=0,
                        help='another GPU for aux training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    train(args)
