import torch
import numpy as np
import path_config as path
import re
from nltk.corpus import brown
import json
import random
import utils
import replace
import get_hot_word as getwords
from models import TextCNN,TextRCNN, TextRNN_Att
from importlib import import_module
import os
from bs4 import BeautifulSoup, NavigableString
import replace
if __name__ == '__main__':
    # 生成对抗样本，找到符合要求的词后，直接在未经处理的banner上改动

    torch.manual_seed(8)
    torch.cuda.manual_seed(8)

    embedding = 'embedding_banner.npz'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'TextCNN'

    x = import_module('models.' + model_name)
    config = x.Config(path.dataset, embedding)
    model = x.Model(config)

    model.load_state_dict(torch.load(path.v_cnn_modelpath,map_location=torch.device('cpu')))

    model = model.to(device)

    dataset = 'dataset-p'
    info = np.load(dataset + '/data/' + embedding, allow_pickle=True)
    embeddings = info['embeddings'][()]
    # torch.manual_seed(8)
    # torch.cuda.manual_seed(8)
    #
    # embedding = 'embedding_banner.npz'
    # # dataset = 'dataset-dt'
    # dataset = 'dataset-dt29'
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # model = TextRNN_Att.Model(TextRNN_Att.Config(dataset, embedding))
    # state = torch.load(path.dt_rnnatt_modelpath, map_location=torch.device('cpu'))
    # model = model.to(device)
    #
    # try:
    #     model.load_state_dict(state['state_dict'])
    # except:
    #     model = torch.nn.DataParallel(model)
    #     model.load_state_dict(state['state_dict'])
    #     model = model.module
    # print('Type input:')
    # info = np.load(dataset + '/data/' + embedding, allow_pickle=True)
    # # print(info.files)
    # embeddings = info['embeddings'][()]
    vocab = utils.vocab


    def predict(input_list):

        input_list1 = utils.raw_preprocessing(input_list)
        input = " ".join(input_list1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #data, label = utils.preprocess(input, device)
        data, label = utils.build_text(input, vocab)
        data = torch.Tensor(data).long().view(1, -1).unsqueeze(dim=0)
        # print(input)
        ori_pred = model(data)  # ori_batch_ent_shaking_outputs
        #self.query_count += 1
        return ori_pred


    def parse_tag(ori_pred, config):
        pred = torch.max(ori_pred.data, 1)[1].cpu().numpy()
        pred = pred[0]
        #device_type, brand, product = config.class_list[pred].split('/')
        brand = config.class_list[pred]
        return [brand]


    def preprocess(input):
        # input -> token list
        text = utils.extracTextFromHtml(input)
        text = text.replace("/", " ").replace("=", " ").replace(":", "").replace("\"", "").replace("'", " ")
        token_list = text.split()
        return token_list
    filelist = os.listdir(path.raw_data_dir)
    # q = 0
    id = 0
    for i, file in enumerate(filelist):
        data_path = path.raw_data_dir + '/' + file
        words_path = path.hotwords_dir_b + '/' + file
        fs_words_path = path.fs_hotwords_dir_b + '/' + file
        processed_data_path = path.processed_data_dir_b + '/' + file
        print('Processing ', data_path)
        other_words = []  # 其它类的 hot words
        all_words = []  # 总重点词表

        # 加载其它类的 hot words
        remaining_filelist = filelist[0:i] + filelist[i + 1:]
        for remaining_file in remaining_filelist:
            other_words.extend(utils.dictlist(path.hotwords_dir_b + '/' + remaining_file))
        other_words_set = set(other_words)
        # 加载本类的 hot words
        self_fs_words = json.loads(open(fs_words_path, 'r', encoding='utf-8').read())
        self_words = utils.dictlist(words_path)
        self_words_set = set(self_words)
        # all_words.extend(list(other_words_set))
        all_words.extend(list(self_words_set))
        # 求出替换时的备选 hot words
        alternative_words = list(other_words_set.difference(self_words_set))
        # brown语料库，其中是常用的有意义的英文词汇
        word_list = brown.words()
        word_set = set(word_list)
        # 正则，匹配数字和字符结合的词
        name_pattern = r"[0-9]+[a-zA-Z]+[0-9a-zA-Z]*|[a-zA-Z]+[0-9]+[0-9a-zA-Z]*"
        # 打开某类数据的json文件（文件中是未处理的数据）单独为某一类生成样本，类别多时比较难操作
        with open(data_path, 'r', encoding='utf-8') as f:
            with open(processed_data_path, 'w', encoding='utf-8') as f2:
                for line in f.readlines():
                    dict = json.loads(line)
                    # content = repr(dict['banner'])
                    input = dict['banner'].lower()  # lower
                    #input1 = preprocess(input)
                    x_p = predict(input)
                    x_ori = parse_tag(x_p, config)