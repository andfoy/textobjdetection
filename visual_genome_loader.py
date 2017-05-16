
import os
import torch
import errno
import os.path as osp
from PIL import Image
import torch.utils.data as data
import visual_genome.local as vg


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


class VisualGenomeLoader(data.Dataset):
    data_path = 'data'
    processed_folder = 'processed'
    corpus_filename = 'corpus.pt'
    region_file = 'region_descriptions.pt'

    def __init__(self, root, transform=None, target_transform=None,
                 train=False, test=False):
        self.root = root
        self.transform = transform
        self.cache = {}

        if not osp.exists(self.root):
            raise RuntimeError('Dataset not found ' +
                               'please download it from: ' +
                               'http://visualgenome.org/api/v0/api_home.html')

        if not self.__check_exists():
            self.process_dataset()

        region_path = osp.join(self.data_path, self.processed_folder,
                               self.region_file)

        corpus_file = osp.join(self.data_path, self.processed_folder,
                               self.corpus_filename)

        with open(region_path, 'rb') as f:
            self.region_descriptions = torch.load(f)

        with open(corpus_file, 'rb') as f:
            self.corpus = torch.load(f)

        # region_descriptions = vg.get_all_region_descriptions(
        #     data_dir=self.root)

    def __check_exists(self):
        processed_path = osp.join(self.data_path, self.processed_folder)
        return osp.exists(processed_path)

    def process_dataset(self):
        # print('Processing scene graphs...')
        # vg.add_attrs_to_scene_graphs(self.root)
        # vg.save_scene_graphs_by_id(data_dir=self.root,
        # image_data_dir=self.graph_path)
        # print('Done!')

        try:
            os.makedirs(os.path.join(self.data_path, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        print("Processing region descriptions...")
        region_descriptions_full = vg.get_all_region_descriptions(
            data_dir=self.root)

        region_descriptions = []
        for region in region_descriptions_full:
            region_descriptions += region

        del region_descriptions_full
        region_path = osp.join(self.data_path, self.processed_folder,
                               self.region_file)

        with open(region_path, 'wb') as f:
            torch.save(region_descriptions, f)

        print("Generating text corpus...")
        corpus = Corpus()
        for i, region in enumerate(region_descriptions):
            print("Processing region: {0}".format(i))
            corpus.add_to_corpus(region.phrase)
            # for region in image_regions:

        corpus.dictionary.add_word('<unk>')

        corpus_file = osp.join(self.data_path, self.processed_folder,
                               self.corpus_filename)

        print("Saving corpus...")
        with open(corpus_file, 'wb') as f:
            torch.save(corpus, f)

        print("Done!")

    def __len__(self):
        return len(self.region_descriptions)

    def __getitem__(self, idx):
        region = self.region_descriptions[idx]
        image_info = region.image

        if image_info.id not in self.cache:
            image_path = image_info.url.split('/')[-2:]
            image_path = osp.join(self.root, *image_path)
            img = Image.open(image_path).convert('RGB')
            self.cache[image_info.id] = img

        img = self.cache[image_info.id]
        img = self.transform(img)

        phrase = self.corpus.tokenize(region.phrase)
        target = torch.LongTensor([region.x, region.y,
                                   region.width, region.height])
        return img, phrase, target
