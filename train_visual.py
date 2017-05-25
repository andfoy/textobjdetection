#!/usr/bin/env python

import os
import time
import torch
import argparse
import torch.nn as nn
import os.path as osp
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from ssd import v2
from ssd.ssd import build_ssd
from lstm_model import RNNModel
from ssd.layers.modules import MultiBoxLoss

from torch.utils.data import DataLoader
from torchvision import transforms, models
from visual_genome_loader import (VisualGenomeLoader,
                                  AnnotationTransform,
                                  ResizeTransform,
                                  detection_collate)

parser = argparse.ArgumentParser(description='Single Shot MultiBox '
                                             'Detector for linguistic object '
                                             'detection Training')
parser.add_argument('--data', type=str, default='../visual_genome',
                    help='path to Visual Genome dataset')
parser.add_argument('--jaccard-threshold', default=0.5, type=float,
                    help='Min Jaccard index for matching')
parser.add_argument('--batch-size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=120000, type=int,
                    help='Number of training epochs')
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--log-iters', action='store_true',
                    help='Print the loss at each iteration')
parser.add_argument('--save-folder', default='weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--rnn-model', type=str, default='LSTM',
                    help='type of recurrent net '
                         '(RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--num-classes', type=int, default=2000,
                    help='number of classification categories')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--lang-model', type=str, default='model2.pt',
                    help='location to LSTM parameters file')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

cfg = v2
num_classes = args.num_classes
ssd_dim = 300
batch_size = args.batch_size

print('Loading train data...')
trainset = VisualGenomeLoader(args.data,
                              transform=transforms.Compose([
                                  ResizeTransform((300, 300)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(
                                      mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]),
                              target_transform=AnnotationTransform())

print('Loading validation data...')
validation = VisualGenomeLoader(args.data,
                                transform=transforms.Compose([
                                    ResizeTransform((300, 300)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]),
                                target_transform=AnnotationTransform(),
                                train=False)

if not osp.exists(args.save_folder):
    os.makedirs(args.save_folder)

net = build_ssd('train', ssd_dim, num_classes)

print('Loading base network...')
vgg = models.vgg16(pretrained=True).state_dict()

state_dict = net.state_dict()
for layer in vgg:
    if layer.startswith('features'):
        _, layer_name = layer.split('features.')
        state_dict['vgg.' + layer_name] = vgg[layer]

# net.load_state_dict(state_dict)

if args.cuda:
    net.cuda()

net.load_state_dict(state_dict)

print('Loading RNN model...')
ntokens = len(trainset.corpus.dictionary)
lang_model = RNNModel(args.rnn_model, ntokens, args.emsize, args.nhid,
                      args.nlayers, args.dropout, args.tied)

if args.cuda:
    lang_model.cuda()

lang_model.eval()

with open(args.lang_model, 'rb') as f:
    state_dict = torch.load(f)
    lang_model.load_state_dict(state_dict)


def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


trainset = DataLoader(trainset, shuffle=True, collate_fn=lambda x:
                      detection_collate(x, lang_model))

print('Initializing weights...')
# initialize newly added layers' weights with xavier method
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)


def train(epoch):
    net.train()
    for img, target, phrase in train:
        pass
        # hidden = lang_model.init_hidden()