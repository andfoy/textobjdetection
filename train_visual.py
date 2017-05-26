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
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
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
parser.add_argument('--num-classes', type=int, default=150,
                    help='number of classification categories')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--lang-model', type=str, default='model2.pt',
                    help='location to LSTM parameters file')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
# parser.add_argument('--top', type=int, default=150,
#                     help='pick top N visual categories')

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
                              target_transform=AnnotationTransform(),
                              top=args.num_classes)

print('Loading validation data...')
validation = VisualGenomeLoader(args.data,
                                transform=transforms.Compose([
                                    ResizeTransform((300, 300)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])]),
                                target_transform=AnnotationTransform(),
                                train=False,
                                top=args.num_classes)

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
                      detection_collate(x, lang_model),
                      batch_size=args.batch_size)

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
    loc_loss = 0
    conf_loss = 0
    total_loss = 0
    start_time = time.time()
    for batch_idx, (imgs, targets, thoughts) in enumerate(trainset):
        if args.cuda:
            imgs = Variable(imgs.cuda())
            targets = [Variable(x.cuda()) for x in targets]
            thoughts = thoughts.cuda()
        optimizer.zero_grad()
        out = net((imgs, thoughts))
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        total_loss += loss[0]
        loc_loss += loss_l[0]
        conf_loss += loss_c[0]

        if batch_idx % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            cur_total_loss = total_loss / args.log_interval
            cur_loc_loss = loc_loss / args.log_interval
            cur_conf_loss = conf_loss / args.log_interval

            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| ms/batch {:.6f} | total loss {:.6f} '
                  '| loc loss {:.6f} | conf loss: {:.6f}'.format(
                      epoch, batch_idx, len(trainset), elapsed_time * 1000,
                      cur_total_loss, cur_loc_loss, cur_conf_loss))

            total_loss = 0
            loc_loss = 0
            conf_loss = 0
            start_time = time.time()


if __name__ == '__main__':
    for epoch in range(1, args.epochs + 1):
        train(epoch)
