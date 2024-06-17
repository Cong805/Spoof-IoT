# -*- coding:utf-8 -*-
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
from models import TextCNN, TextRCNN, TextRNN_Att
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

    model.load_state_dict(torch.load(config.save_path, map_location=torch.device('cpu')))

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
        # data, label = utils.preprocess(input, device)
        data, label = utils.build_text(input, vocab)
        data = torch.Tensor(data).long().view(1, -1).unsqueeze(dim=0)
        # print(input)
        ori_pred = model(data)  # ori_batch_ent_shaking_outputs
        # self.query_count += 1
        return ori_pred


    def parse_tag(ori_pred, config):
        pred = torch.max(ori_pred.data, 1)[1].cpu().numpy()
        pred = pred[0]
        device_type, brand, product = config.class_list[pred].split('/')
        #brand = config.class_list[pred]
        return [device_type, brand, product]


    filelist = os.listdir(path.raw_data_dir)
    # q = 0
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

        cos_sim_words_dict = {}
        with open(path.sim_vocab_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                dic = json.loads(line)
                for word, word_list in dic.items():
                    cos_sim_words_dict[word] = word_list
        # 打开某类数据的json文件（文件中是未处理的数据）单独为某一类生成样本，类别多时比较难操作
        with open(data_path, 'r', encoding='utf-8') as f:
            with open(processed_data_path, 'w', encoding='utf-8') as f2:
                for line in f.readlines():
                    dict = json.loads(line)
                    # content = repr(dict['banner'])
                    input = dict['banner'].lower()  # lower
                    # 提取banner中HTML的内容
                    index_start = 0
                    if re.search('<!doctype html', input, re.IGNORECASE) is None:
                        if re.search('<html>', input, re.IGNORECASE) is None:
                            if re.search('<?xml', input, re.IGNORECASE) is None:
                                index_start = len(input)
                                banner_head = input
                                html_doc = ''
                            else:
                                index_start = re.search('<?xml', input, re.IGNORECASE).span()[0]
                                if re.search('</html>', input, re.IGNORECASE):
                                    index_end = re.search('</html>', input, re.IGNORECASE).span()[1]
                                    banner_head = input[0:index_start]  # 不可见部分的一部分
                                    html_doc = input[index_start:index_end]
                                    del index_end
                                else:
                                    # index_end = re.search('</html>', input, re.IGNORECASE).span()[1]
                                    banner_head = input[0:index_start]  # 不可见部分的一部分
                                    html_doc = input[index_start:]


                        else:
                            index_start = re.search('<html>', input, re.IGNORECASE).span()[0]
                            if re.search('</html>', input, re.IGNORECASE):
                                index_end = re.search('</html>', input, re.IGNORECASE).span()[1]
                                banner_head = input[0:index_start]  # 不可见部分的一部分
                                html_doc = input[index_start:index_end]
                                del index_end
                            else:
                                # index_end = re.search('</html>', input, re.IGNORECASE).span()[1]
                                banner_head = input[0:index_start]  # 不可见部分的一部分
                                html_doc = input[index_start:]

                    else:
                        index_start = re.search('<!doctype html', input, re.IGNORECASE).span()[0]
                        if re.search('</html>', input, re.IGNORECASE):
                            index_end = re.search('</html>', input, re.IGNORECASE).span()[1]
                            banner_head = input[0:index_start]  # 不可见部分的一部分
                            html_doc = input[index_start:index_end]
                            del index_end
                        else:
                            # index_end = re.search('</html>', input, re.IGNORECASE).span()[1]
                            banner_head = input[0:index_start]  # 不可见部分的一部分
                            html_doc = input[index_start:]

                    # js_list = re.findall('<script.*?>[\s\S]*?</script>', html_doc, flags=re.IGNORECASE)

                    # if re.search('<!--[^>]*-->', html_doc):
                    #     html_doc = re.sub('<!--[^>]*-->', '', html_doc)
                    # 提取css和script中的注释/* */
                    note_start = re.finditer('/\*', html_doc)
                    css_script_note = list()
                    for i in note_start:
                        s_note = i.span()[0]
                        e_note = html_doc.find('*/', s_note)
                        css_script_note.append(html_doc[s_note + 2:e_note])

                        # 将注释的内容设为空
                        del s_note, e_note
                    for item in css_script_note:
                        s = "/*" + item + "*/"
                        html_doc = html_doc.replace(s, "")

                    js_list = re.findall('<script.*?>[\s\S]*?</script>', html_doc, flags=re.IGNORECASE)
                    for i in range(len(js_list)):
                        html_doc = re.sub(re.escape(js_list[i]), r'~~~', html_doc)  # 所有的js代码要保持不变

                    # text = utils.raw_preprocessing(html_doc)
                    # texts = ' '.join(text)
                    # texts = banner_head + texts
                    text = utils.raw_preprocessing(input)
                    texts = ' '.join(text)
                    # print(texts)
                    ori_text_list = utils.ori_preprocessing(input)
                    count = 0
                    vis_words = {}
                    invis_words = {}
                    perturbable_words_id = []
                    perturbable_words_substitution = {}
                    x_new = ori_text_list
                    for index, token in enumerate(ori_text_list):
                        if token in cos_sim_words_dict.keys():
                            perturbable_words_id.append(index)
                            perturbable_words_substitution[index] = cos_sim_words_dict[token]
                    # 选取text即原始输入数据中在重点词表中的数据
                    for i in range(0,20):
                        if len(perturbable_words_id) > 0:
                            rand_idx = np.random.choice(len(perturbable_words_id), 1)[0]
                            id = perturbable_words_id[rand_idx]
                            while x_new[id] != ori_text_list[id]:
                                rand_idx = np.random.choice(len(perturbable_words_id), 1)[0]
                            id = perturbable_words_id[rand_idx]
                            replace_list = perturbable_words_substitution[id]
                            # 从词表中随机选择一个词
                            rand_replace_id = np.random.choice(len(replace_list), 1)[0]
                            # 如果选取的可替换词和目前样本中的数据相同并且还有可以替换的词，重新选择
                            while x_new[rand_idx] == replace_list[rand_replace_id] and len(replace_list) > 1:
                                rand_replace_id = np.random.choice(len(replace_list), 1)[0]

                            banner_head = re.sub(ori_text_list[rand_idx], replace_list[rand_replace_id], banner_head,
                                                 flags=re.IGNORECASE)
                            invis_words.update({ori_text_list[rand_idx]: replace_list[rand_replace_id]})
                        # 确定要改变的词得到一个样本

                    for i in range(len(js_list)):
                        # print(js_list[i])
                        # content = re.sub(r'~~~', js_list[i].replace('\\', '/'), content, count=1)  # .replace('\\', '/')
                        html_doc = re.sub(r'~~~', js_list[i].replace('\\', '/'), html_doc, count=1)
                    content = banner_head + html_doc

                    standard_dict = {
                        #'succ': succ,
                        'device_type': dict['device_type'],
                        'brand': dict['brand'],
                        'product': dict['product'],
                        #'adv_product':x_tag[2],
                        'change_words': invis_words,
                        "sum_all": len(invis_words),
                        'banner': dict['banner'],
                        'adv_banner': content
                    }
                    json.dump(standard_dict, f2)
                    f2.write('\n')
                    del banner_head
                    del html_doc
                    del content
                    del dict
                    del standard_dict, index_start, js_list, css_script_note, note_start, texts, text, count, input, ori_text_list
        del all_words, other_words, other_words_set, self_words, self_words_set, alternative_words
    # print(q/3000)