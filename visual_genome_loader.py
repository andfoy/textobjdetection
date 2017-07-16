
import os
import cv2
import json
import torch
import errno
import codecs
import progressbar
import numpy as np
import os.path as osp
from PIL import Image
import torch.utils.data as data
import visual_genome.local as vg
from torch.autograd import Variable


def detection_collate(batch, rnnmodel):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image
                                 are stacked on 0 dim
    """
    targets = []
    imgs = []
    thoughts = []
    for sample in batch:
        idx, img, target, phrases = sample
        imgs.append(img)
        targets.append(torch.stack([torch.Tensor(a) for a in target], 0))
        for i in range(0, len(phrases)):
            phrase = phrases[i]
            phrase_col = phrase.view(phrase.size(0), -1)
            hidden = rnnmodel.init_hidden(phrase_col.size(1))
            _, hidden = rnnmodel(Variable(phrase_col.cuda()), hidden)
            hidden = torch.stack(hidden, 0).view(-1, 1).data
            thoughts.append(hidden)
    return torch.stack(imgs, 0), targets, torch.stack(thoughts, 0)


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self):
        self.dictionary = Dictionary()

    def add_to_corpus(self, line):
        """Tokenizes a text line."""
        # Add words to the dictionary
        words = line.split() + ['<eos>']
        # tokens = len(words)
        for word in words:
            self.dictionary.add_word(word)

    def tokenize(self, line):
        # Tokenize line contents
        words = line.split() + ['<eos>']
        tokens = len(words)
        ids = torch.LongTensor(tokens)
        token = 0
        for word in words:
            if word not in self.dictionary.word2idx:
                word = '<unk>'
            ids[token] = self.dictionary.word2idx[word]
            token += 1

        return ids

    def tokenize_file(self, file_path):
        tokens = []
        with codecs.open(file_path, 'r', 'utf-8') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    if word not in self.dictionary.word2idx:
                        word = '<unk>'
                    token = self.dictionary.word2idx[word]
                    tokens.append(token)
        return torch.LongTensor(tokens)


class AnnotationTransform(object):
    def __call__(self, regions, corpus, region_objects,
                 objects_idx, height, width):
        phrases = []
        bboxes = []
        for region in regions:
            try:
                reg_obj = region_objects[region.image.id][region.id]
                reg_obj = frozenset([x.lower()
                                     for x in reg_obj])
            except KeyError:
                reg_obj = frozenset({})
            if reg_obj in objects_idx:
                cat = objects_idx[reg_obj]
                # x_max = min(region.x + region.width, width)
                # y_max = min(region.y + region.height, height)
                x_max = region.x + region.width
                y_max = region.y + region.height

                bbx = [region.x / width, region.y / height,
                       x_max / width,
                       y_max / height,
                       cat]
                bboxes.append(bbx)
                phrases.append(corpus.tokenize(region.phrase))
        return bboxes, phrases


class ResizeTransform(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize(self.size, self.interpolation)


class VisualGenomeLoader(data.Dataset):
    data_path = 'data'
    processed_folder = 'processed'
    corpus_file = 'corpus.pt'
    train_text_file = 'train.txt'
    val_text_file = 'val.txt'
    test_text_file = 'test.txt'
    region_train_file = 'region_train.pt'
    region_val_file = 'region_val.pt'
    region_test_file = 'region_test.pt'
    region_objects_file = 'region_objects.pt'
    obj_idx_file = 'obj_idx.pt'
    human_cat = frozenset({'man', 'woman', 'men', 'women', 'person',
                           'people', 'human', 'lady', 'ladies',
                           'guy', 'guys', 'boy', 'girl', 'boys',
                           'girls', 'pedestrian', 'passenger'})

    def __init__(self, root, transform=None, target_transform=None,
                 train=True, test=False, top=100, group=True,
                 additional_transform=None):
        self.root = root
        self.transform = transform
        self.additional_transform = additional_transform
        self.target_transform = target_transform
        self.top_objects = top
        self.top_folder = 'top_{0}'.format(top)
        self.group = group

        if not osp.exists(self.root):
            raise RuntimeError('Dataset not found ' +
                               'please download it from: ' +
                               'http://visualgenome.org/api/v0/api_home.html')

        if not self.__check_exists():
            self.process_dataset()

        # self.region_objects, self.obj_idx = self.load_region_objects()

        if train:
            train_file = osp.join(self.data_path, self.top_folder,
                                  self.region_train_file)
            with open(train_file, 'rb') as f:
                self.regions = torch.load(f)
        elif test:
            test_file = osp.join(self.data_path, self.top_folder,
                                 self.region_test_file)
            with open(test_file, 'rb') as f:
                self.regions = torch.load(f)
        else:
            val_file = osp.join(self.data_path, self.top_folder,
                                self.region_val_file)
            with open(val_file, 'rb') as f:
                self.regions = torch.load(f)

        if self.group:
            self.regions = self.__group_regions_by_id(self.regions)

        corpus_file = osp.join(self.data_path, self.processed_folder,
                               self.corpus_file)
        with open(corpus_file, 'rb') as f:
            self.corpus = torch.load(f)

        region_obj_file = osp.join(self.data_path, self.top_folder,
                                   self.region_objects_file)
        with open(region_obj_file, 'rb') as f:
            self.region_objects = torch.load(f)

        obj_idx_path = osp.join(self.data_path, self.top_folder,
                                self.obj_idx_file)

        with open(obj_idx_path, 'rb') as f:
            self.obj_idx = torch.load(f)

        self.idx_obj = {v: k for k, v in self.obj_idx.items()}
        # del region_objects

    def __load_region_objects(self):
        print("Loading region objects...")
        region_graph_file = osp.join(self.root, 'region_graphs.json')
        with open(region_graph_file, 'r') as f:
            reg_graph = json.load(f)

        print("Processing regions...")
        img_id = {x['image_id']: {y['region_id']: set([z['entity_name']
                                                       for z in y['synsets']] +
                                                      [z['name']
                                                       for z in y['objects']])
                                  for y in x['regions']}
                  for x in reg_graph}

        # obj_idx = {}
        print("Filtering top {0} human categories...".format(self.top_objects))
        obj_count = {}
        bar = progressbar.ProgressBar()
        for img in bar(img_id):
            for region in img_id[img]:
                obj = frozenset([x.lower() for x in img_id[img][region]])
                if len(obj & self.human_cat) > 0:
                    if obj not in obj_count:
                        obj_count[obj] = 0
                    obj_count[obj] += 1

        top_objs = sorted(obj_count, key=lambda k: obj_count[k],
                          reverse=True)[:self.top_objects]
        obj_idx = {top_objs[i]: i for i in range(0, len(top_objs))}
        # del obj_count
        return img_id, obj_idx

    def __group_regions_by_id(self, regions):
        print("Transforming data....")
        regions_img = {}
        for region in regions:
            if region.image.id not in regions_img:
                regions_img[region.image.id] = []
            regions_img[region.image.id].append(region)
        return list(regions_img.values())

    def __filter_regions_by_class(self, regions):
        print("Filtering regions...")
        act_regions = []
        region_sub = {}
        bar = progressbar.ProgressBar()
        for region in bar(regions):
            try:
                reg_obj = self.region_objects[region.image.id][region.id]
                reg_obj = frozenset([x.lower()
                                     for x in reg_obj])
            except KeyError:
                reg_obj = frozenset({})

            if reg_obj in self.obj_idx:
                act_regions.append(region)
                if region.image.id not in region_sub:
                    region_sub[region.image.id] = {}
                reg_img = region_sub[region.image.id]
                global_region_img = self.region_objects[region.image.id]
                reg_img[region.id] = global_region_img[region.id]
        return act_regions, region_sub

    def __check_exists(self):
        path = osp.join(self.data_path, self.top_folder)
        return osp.exists(path)

    def process_dataset(self):
        try:
            os.makedirs(osp.join(self.data_path, self.top_folder))
            os.makedirs(osp.join(self.data_path, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # print("Generating top images set...")
        # img_top_ids = self.get_top_images()
        self.region_objects, self.obj_idx = self.__load_region_objects()

        print("Processing region descriptions...")
        region_descriptions_full = vg.get_all_region_descriptions(
            data_dir=self.root)

        region_descriptions = []
        for region in region_descriptions_full:
            region_descriptions += region

        # del region_descriptions_full

        corpus_path = osp.join(self.data_path, self.processed_folder,
                               self.corpus_file)

        if not osp.exists(corpus_path):
            print("Generating text corpus...")
            corpus = Corpus()
            for i, region in enumerate(region_descriptions):
                print("Processing region: {0}".format(i))
                corpus.add_to_corpus(region.phrase)

            corpus.dictionary.add_word('<unk>')
            print("Saving corpus to file...")
            with open(corpus_path, 'wb') as f:
                torch.save(corpus, f)

        # print("Selecting region descriptions from top images...")
        # regions = []
        # bar = progressbar.ProgressBar()
        # for region in bar(region_descriptions_full):
        #     # print("Processing region: {0}".format(i))
        #     if region[0].image.id in img_top_ids:
        #         regions += region
        regions, regions_objects = self.__filter_regions_by_class(
            region_descriptions)

        print("Splitting region descriptions...")
        train_prop = int(np.ceil(len(regions) * 0.6))
        val_train_prop = int(np.ceil(len(regions) * 0.15))

        regions = np.array(regions)
        np.random.shuffle(regions)

        train_regions = regions[:train_prop].tolist()
        val_regions = regions[train_prop:train_prop + val_train_prop].tolist()
        test_regions = regions[train_prop + val_train_prop:].tolist()

        print("Saving train text corpus...")
        train_text_path = osp.join(self.data_path, self.top_folder,
                                   self.train_text_file)
        with codecs.open(train_text_path, 'w', 'utf-8') as f:
            for region in train_regions:
                f.write(region.phrase + '\n')

        print("Saving validation text corpus...")
        val_text_path = osp.join(self.data_path, self.top_folder,
                                 self.val_text_file)
        with codecs.open(val_text_path, 'w', 'utf-8') as f:
            for region in val_regions:
                f.write(region.phrase + '\n')

        print("Saving test text corpus...")
        test_text_path = osp.join(self.data_path, self.top_folder,
                                  self.test_text_file)
        with codecs.open(test_text_path, 'w', 'utf-8') as f:
            for region in test_regions:
                f.write(region.phrase + '\n')

        print("Saving training regions...")
        train_file = osp.join(self.data_path, self.top_folder,
                              self.region_train_file)
        with open(train_file, 'wb') as f:
            torch.save(train_regions, f)

        print("Saving validation regions...")
        val_file = osp.join(self.data_path, self.top_folder,
                            self.region_val_file)
        with open(val_file, 'wb') as f:
            torch.save(val_regions, f)

        print("Saving testing regions...")
        test_file = osp.join(self.data_path, self.top_folder,
                             self.region_test_file)
        with open(test_file, 'wb') as f:
            torch.save(test_regions, f)

        print("Saving dataset objects per region...")
        regions_obj_file = osp.join(self.data_path, self.top_folder,
                                    self.region_objects_file)
        with open(regions_obj_file, 'wb') as f:
            torch.save(regions_objects, f)

        print("Saving object to index map...")
        obj_idx_path = osp.join(self.data_path, self.top_folder,
                                self.obj_idx_file)
        with open(obj_idx_path, 'wb') as f:
            torch.save(self.obj_idx, f)

        print("Done!")

    def group_class_img_bbx(self):
        class_img_bbx = {}
        regions = self.regions
        if self.group:
            regions = []
            for img_regions in self.regions:
                regions += img_regions
            # regions = self.__group_regions_by_id(self.regions)
        for region in regions:
            try:
                reg_obj = self.region_objects[region.image.id][region.id]
                reg_obj = frozenset([x.lower()
                                     for x in reg_obj])
            except KeyError:
                reg_obj = frozenset({})
            if reg_obj in self.obj_idx:
                cat = self.obj_idx[reg_obj]
                if cat not in class_img_bbx:
                    class_img_bbx[cat] = {}
                if region.image.id not in class_img_bbx[cat]:
                    class_img_bbx[cat][region.image.id] = []
                x_max = region.x + region.width
                y_max = region.y + region.height

                bbx = [region.x, region.y,
                       x_max,
                       y_max]
                class_img_bbx[cat][region.image.id].append(bbx)
        return class_img_bbx

    def get_top_images(self):
        obj_file_path = osp.join(self.root, 'objects.json')
        objects = json.load(open(obj_file_path, 'r'))

        total_objects = {}
        for img_obj in objects:
            for obj in img_obj['objects']:
                for name in obj['names']:
                    if name not in total_objects:
                        total_objects[name] = 0
                    total_objects[name] += 1

        sorted_objs = sorted(total_objects.keys(),
                             key=lambda k: total_objects[k],
                             reverse=True)
        sorted_objs = sorted_objs[0:self.top_objects]

        valid_img_ids = []
        for img_obj in objects:
            found = False
            for obj in img_obj['objects']:
                for name in obj['names']:
                    if name in sorted_objs:
                        valid_img_ids.append(img_obj['image_id'])
                        found = True
                        break
                if found:
                    break

        return valid_img_ids

    def pull_image(self, idx):
        regions = self.regions[idx]
        if self.group:
            image_info = regions[0].image
        else:
            image_info = regions.image
        image_path = image_info.url.split('/')[-2:]
        image_path = osp.join(self.root, *image_path)
        return np.array(Image.open(image_path).convert('RGB'))

    def pull_anno(self, idx):
        regions = self.regions[idx]
        bboxes = []
        phrases = []

        if not self.group:
            regions = [regions]

        for region in regions:
            bbx = [region.x, region.y,
                   (region.x + region.width),
                   (region.y + region.height)]
            bboxes.append(bbx)
            phrases.append(region.phrase)
        return bboxes, phrases

    def __len__(self):
        return len(self.regions)

    def __getitem__(self, idx):
        regions = self.regions[idx]
        if not self.group:
            regions = [regions]

        image_info = regions[0].image

        image_path = image_info.url.split('/')[-2:]
        image_path = osp.join(self.root, *image_path)

        # img = Image.open(image_path).convert('RGB')
        img = cv2.imread(image_path)

        bboxes, phrases = self.target_transform(regions,
                                                self.corpus,
                                                self.region_objects,
                                                self.obj_idx,
                                                image_info.height,
                                                image_info.width)

        if self.transform is not None:
            bboxes = np.array(bboxes)
            img, boxes, labels = self.transform(img, bboxes[:, :4],
                                                bboxes[:, 4])
            # img = img.transpose(2, 0, 1)
            bboxes = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        img = img[:, :, (2, 1, 0)]
        img = Image.fromarray(img)
        print(type(img))
        print(img.shape)
        # img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.additional_transform(img)

        return image_info.id, img, bboxes, phrases
