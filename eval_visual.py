"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torchvision.transforms as transforms
# from torch.autograd import Variable
# from data import VOCroot
# from data import VOC_CLASSES as labelmap
# import torch.utils.data as data

# from data import AnnotationTransform, VOCDetection, BaseTransform
# from ssd import build_ssd

# import sys
# import os
# import time
# import argparse
# import numpy as np
# import pickle
# import cv2

# if sys.version_info[0] == 2:
#     import xml.etree.cElementTree as ET
# else:
#     import xml.etree.ElementTree as ET

import os
import time
import torch
import pickle
import argparse
import numpy as np
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

# from ssd.data import BaseTransform
from torchvision import transforms
from torch.utils.data import DataLoader
from visual_genome_loader import (VisualGenomeLoader,
                                  AnnotationTransform,
                                  ResizeTransform,
                                  detection_collate)

parser = argparse.ArgumentParser(description='Linguistic Single Shot MultiBox'
                                             ' Detection')
parser.add_argument('--data', type=str, default='../visual_genome',
                    help='path to Visual Genome dataset')
parser.add_argument('--trained_model', default='weights/ssd_lang.pt',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--rnn-model', type=str, default='LSTM',
                    help='type of recurrent net '
                         '(RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='humber of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--num-classes', type=int, default=150,
                    help='number of classification categories')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--lang-model', type=str, default='model3.pt',
                    help='location to LSTM parameters file')
parser.add_argument('--lang', action='store_true',
                    help='test SSD model with language features')


args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

cfg = v2
num_classes = args.num_classes
ssd_dim = 300
# batch_size = args.batch_size
group = not args.lang
set_type = 'test'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} VOC results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dataset.ids):
                dets = all_boxes[cls_ind+1][im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(index, dets[k, -1],
                                   dets[k, 0], dets[k, 1],
                                   dets[k, 2], dets[k, 3]))


def do_python_eval(box_list, dataset, output_dir='output', use_07=True):
    # cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = use_07
    # print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    ground_truth_list = dataset.group_class_img_bbx()
    for i, cls in enumerate(dataset.obj_idx):
        # filename = get_voc_results_file_template(set_type, cls)
        # rec, prec, ap = voc_eval(
           # filename, annopath, imgsetpath.format(set_type), cls, cachedir,
           # ovthresh=0.5, use_07_metric=use_07_metric)
        if cls not in box_list:
            rec, prec, ap = -1, -1, -1
        else:
            rec, prec, ap = vg_eval(box_list[cls], ground_truth_list[cls])
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def vg_eval(class_box_list, ground_truth_list, ovthresh=0.5,
            use_07_metric=True):
    # complete_info = []
    # for img_id in class_box_list:
    #     complete_info += class_box_list[img_id]
    # complete_info = np.array(complete_info)

    # confidence = complete_info[:, -1]
    # BB
    nd = len(ground_truth_list)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d, img_id in enumerate(ground_truth_list):
        BBGT = np.array(ground_truth_list[img_id])
        pred_info = np.array(class_box_list[img_id])
        BB = pred_info[:, :-1]
        confidence = pred_info[:, -1]

        sorted_ind = np.argsort(-confidence)
        bb = BB[sorted_ind, :]
        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih

            # # union
            # uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
            #        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
            #        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
            uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                   (BBGT[:, 2] - BBGT[:, 0]) *
                   (BBGT[:, 3] - BBGT[:, 1]) - inters)
            overlaps = inters / uni
            ovmax = np.max(overlaps)
            # jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            tp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(len(ground_truth_list))
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default False)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih

                # # union
                # uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                #        (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                #        (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net_lang(save_folder, net, cuda, dataset, transform, top_k,
                  im_size=300, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    # all_boxes = [[[] for _ in range(num_images)]
    # for _ in range(len(labelmap)+1)]

    all_boxes = {}
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_lang', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im = dataset.pull_image(i)
        scale = torch.Tensor([im.shape[1], im.shape[0],
                              im.shape[1], im.shape[0]])
        img_id, img, bboxes, phrases = dataset[i]
        x, targets, thoughts = transform([(img_id, img, bboxes, phrases)])

        if args.cuda:
            x = Variable(x.cuda())
            thoughts = Variable(thoughts.cuda())
        _t['im_detect'].tic()

        # if args.lang:
        x = (x, thoughts)
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        det_class = bboxes[0][-1]

        # skip j = 0, because it's the background class
        # for j in range(0, detections.size(1)):
        dets = detections[0, det_class, :]
        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        if dets.dim() == 0:
            continue
        boxes = dets[:, 1:]
        scores = dets[:, 0].cpu().numpy()
        boxes *= scale.unsqueeze(0).expand_as(boxes)
        cls_dets = np.hstack((boxes.cpu().numpy(),
                              scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        text_class = dataset.idx_obj[det_class]
        if text_class not in all_boxes:
            all_boxes[text_class] = {}
        # if img_id not in all_boxes[text_class]:
        #     all_boxes[text_class][img_id] = []
        all_boxes[text_class][img_id] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    evaluate_detections(all_boxes, output_dir, dataset)


def evaluate_detections(box_list, output_dir, dataset):
    # write_voc_results_file(box_list, dataset)
    do_python_eval(box_list, dataset, output_dir)


if __name__ == '__main__':
    # load net
    net = build_ssd('test', 300, args.num_classes)    # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    # dataset = VOCDetection(VOCroot, set_type, None, AnnotationTransform())
    print('Loading test data...')
    testset = VisualGenomeLoader(args.data,
                                 transform=transforms.Compose([
                                     ResizeTransform((300, 300)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]),
                                 target_transform=AnnotationTransform(),
                                 test=True,
                                 top=args.num_classes,
                                 group=group)
    print('Loading RNN model...')
    ntokens = len(testset.corpus.dictionary)
    lang_model = RNNModel(args.rnn_model, ntokens, args.emsize, args.nhid,
                          args.nlayers, args.dropout, args.tied)

    if args.cuda:
        lang_model.cuda()

    lang_model.eval()

    with open(args.lang_model, 'rb') as f:
        state_dict = torch.load(f)
        lang_model.load_state_dict(state_dict)

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    if args.lang:
        test_net_lang(args.save_folder, net, args.cuda, testset,
                      lambda x: detection_collate(x, lang_model),
                      args.top_k, 300, thresh=args.confidence_threshold)
