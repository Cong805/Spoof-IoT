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
from models import TextCNN,TextRCNN, TextRNN_Att
from importlib import import_module
import os
from bs4 import BeautifulSoup, NavigableString
import replace

# (word_index, model, texts, embeddings,vocab,word, banner_head, alternative_words)
def region_repalce(word_index, model, texts, embeddings,vocab,word, content, alternative_words):
    soup = BeautifulSoup(content, 'html.parser')
    #meta
    invis = 0
    vis = 0
    sum_twords = []
    sum_twords_invis = []
    def invisible_meta_replace(soup, meta, meta_words):
        twords = []
        if word in meta_words:
            #找相似性的词替换
            # meta['content'] = re.sub(word, replace.visual(word, replace.similar_0), meta_words, flags=re.IGNORECASE)
            # twords = replace.most_largest_cos_other(embeddings, vocab, word, alternative_words)
            twords = replace.replace_insert_sim(embeddings, vocab, word, alternative_words)
            # twords = replace.find_least_leven(word, alternative_words)
            # twords = replace.invisible_insert1(word)
            # tword = replace.find_most_similiar(word, alternative_words)
            # if tword:  # next(iter(hotword)
            #     meta['content'] = re.sub(word, next(iter(tword)), meta_words, flags=re.IGNORECASE)
            # else:
            #     meta['content'] = re.sub(word, replace.visual(word, replace.similar_0), meta_words, flags=re.IGNORECASE)  # 还可以再加参数限制改几个
            meta['content'] = re.sub(word, twords, meta_words, flags=re.IGNORECASE)

        return twords,str(soup) #.prettify()

    for meta in soup.find_all('meta'):
        # new_meta = meta
        if meta.get('name') is None:
            continue
        if meta["name"].lower() == "keywords":
            meta_words = meta['content']
            # meta中替换，然后再放回
            a,content = invisible_meta_replace(soup, meta, meta_words)
            if len(a) >=1:
                sum_twords.append(a)
                invis = invis +1
        if meta["name"].lower() == "description":
            meta_words = meta['content']
            a,content = invisible_meta_replace(soup, meta, meta_words)
            if len(a) >=1:
                sum_twords.append(a)
                invis = invis + 1
        if meta["name"].lower() == "generator":
            meta_words = meta['content']
            a,content = invisible_meta_replace(soup, meta, meta_words)
            if len(a) >=1:
                sum_twords.append(a)
                invis = invis + 1
        if meta["name"].lower() == "author":
            meta_words = meta['content']
            a,content = invisible_meta_replace(soup, meta, meta_words)
            if len(a) >=1:
                sum_twords.append(a)
                invis = invis + 1

    # level 1

    def similar_replace(soup,head, text):
        twords = []
        if word in text:
            # text = re.sub(word, replace.visual(word, replace.similar_0), text, flags=re.IGNORECASE)
            tword = replace.invisible_transpose_replace1(embeddings, vocab, word, alternative_words)
            twords.append(tword)
            text = re.sub(word, tword, text, flags=re.IGNORECASE)
            head.string = text
        return twords,str(soup)

    for head in soup.find_all("h1"):
        text = head.get_text()  # text要先完成替换然后再放回原来的地方
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis +1
    for head in soup.find_all("h2"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("h3"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("h4"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("h5"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("h6"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("b"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("big"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("em"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("strong"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("i"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for head in soup.find_all("u"):
        text = head.get_text()
        b,content = similar_replace(soup, head, text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1
    for title_items in soup.find_all("title"):
        title_text = title_items.get_text()
        b,content = similar_replace(soup, title_items, title_text)
        if len(b) >= 1:
            sum_twords.append(b)
            vis = vis + 1

    #level2
    def level2_replace(soup, items, text):
        twords = []
        if word in text:
            # tword = replace.most_largest_cos_other_sim(embeddings, vocab, word, alternative_words)
            # tword = replace.invisible_transpose_replace1(embeddings, vocab, word, alternative_words)
            tword = replace.invisible_transpose_replace1(embeddings, vocab, word, alternative_words)
            twords.append(tword)
            # text = re.sub(re.escape(word), replace.visual(word, replace.similar_0), text, flags=re.IGNORECASE)
            # tword = replace.find_most_similiar(word, alternative_words)
            # if tword:
            #     text = re.sub(re.escape(word), next(iter(tword)), text, flags=re.IGNORECASE)
            # else:
            #     text = re.sub(re.escape(word), replace.visual(word, replace.similar_0), text, flags=re.IGNORECASE)
            # tword = replace.find_least_leven(word, alternative_words)
            # tword = replace.invisible_tramspose(word)
            text = re.sub(re.escape(word), tword, text, flags=re.IGNORECASE)
            items.string = text
        return twords,str(soup)

    for a_items in soup.find_all("a"):
        a_text = a_items.get_text()
        c,content = level2_replace(soup, a_items, a_text)
        if len(c) >= 1:
            sum_twords.append(c)
            vis = vis + 1
    for li_items in soup.select("ul > li"):
        li_text = li_items.get_text()
        c,content = level2_replace(soup, li_items, li_text)
        if len(c) >= 1:
            sum_twords.append(c)
            vis = vis + 1
    for li_items in soup.select("ol > li"):
        li_text = li_items.get_text()
        c,content = level2_replace(soup, li_items, li_text)
        if len(c) >= 1:
            sum_twords.append(c)
            vis = vis + 1
    for dt_items in soup.select("dl > dt"):
        dt_text = dt_items.get_text()
        c,content = level2_replace(soup, dt_items, dt_text)
        if len(c) >= 1:
            sum_twords.append(c)
            vis = vis + 1
    for dd_items in soup.select("dl > dd"):
        dd_text = dd_items.get_text()
        c,content = level2_replace(soup, dd_items, dd_text)
        if len(c) >= 1:
            sum_twords.append(c)
            vis = vis + 1
    for option_items in soup.select("select > option"):
        option_text = option_items.get_text()
        c,content = level2_replace(soup, option_items, option_text)
        if len(c) >= 1:
            sum_twords.append(c)
            vis = vis + 1

    #level3
    def level3_replace(soup, items, text):
        twords = []
        if word in text:
            # tword = replace.find_most_similiar(word, alternative_words)
            # if tword:  # next(iter(hotword)
            #     text = re.sub(re.escape(word), next(iter(tword)), text, flags=re.IGNORECASE)
            # else:
            #     text = re.sub(re.escape(word), replace.visual(word, replace.similar_0), text,
            #                   flags=re.IGNORECASE)  # 还可以再加参数限制改几个
            # text = re.sub(word, replace.visual(word, replace.similar_0), text, flags=re.IGNORECASE)
            # tword = replace.find_least_leven(word, alternative_words)
            # tword = replace.visual_replace1(embeddings, vocab, word, alternative_words)
            tword = replace.visual_replace1(embeddings, vocab, word, alternative_words)
            twords.append(tword)
            # tword = replace.invisible_tramspose(word)
            # tword = replace.most_largest_cos_other_sim(embeddings, vocab, word, alternative_words)
            text = re.sub(re.escape(word), tword, text, flags=re.IGNORECASE)
            items.string = text
        return twords,str(soup)


    for ta_items in soup.select("table > caption"):
        ta_text = ta_items.get_text()
        d,content = level3_replace(soup, ta_items, ta_text)
        if len(d) >= 1:
            sum_twords.append(d)
            vis = vis + 1

    for ta_items in soup.select("table > tr > th"):
        ta_text = ta_items.get_text()
        d,content = level3_replace(soup, ta_items, ta_text)
        if len(d) >= 1:
            sum_twords.append(d)
            vis = vis + 1
    for ta_items in soup.select("table > tr > td"):
        ta_text = ta_items.get_text()
        d,content = level3_replace(soup, ta_items, ta_text)
        if len(d) >= 1:
            sum_twords.append(d)
            vis = vis + 1
    # level4如果不在明显的地方，则找其它类重点词中，拼写相近的替换，如果找不到，再用相似字符替换
    def level4_replace(soup, items, text):
        twords = []
        if word in text:
            # text = re.sub(word, replace.visual(word, replace.similar_0), text, flags=re.IGNORECASE)
            # tword = replace.find_most_similiar(word, alternative_words)
            # if tword:  # next(iter(hotword)
            #     text = re.sub(re.escape(word), next(iter(tword)), text, flags=re.IGNORECASE)
            # else:
            #     text = re.sub(re.escape(word), replace.visual(word, replace.similar_0), text,
            #                      flags=re.IGNORECASE)  # 还可以再加参数限制改几个
            # tword = replace.find_least_leven(word, alternative_words)
            # tword = replace.most_largest_cos_other_sim(embeddings, vocab, word, alternative_words)
            tword = replace.cossim_replace1(embeddings, vocab, word, alternative_words)
            twords.append(tword)
            # tword = replace.invisible_tramspose(word)
            text = re.sub(re.escape(word), tword, text, flags=re.IGNORECASE)
            items.string = text
        return twords,str(soup)

    if soup.find_all("body"):
        a = soup.body.descendants
        for i in a:
            if not isinstance(i, NavigableString):
                i_text = i.get_text()
                e,content = level4_replace(soup, i, i_text)
                #print(content)
                if len(e) >= 1:
                    sum_twords_invis.append(e)
                    invis = invis + 1
    else:
        # hotword = replace.find_least_leven(word, alternative_words)
        # hotword = replace.most_largest_cos_other(embeddings, vocab, word, alternative_words)
        hotword = replace.replace_insert_sim(embeddings, vocab, word, alternative_words)
        if len(hotword) >= 1:
            sum_twords_invis.append(hotword)
            invis = invis+1
        #sum_twords.append(hotword)
        # hotword = replace.invisible_tramspose(word)
        content = re.sub(re.escape(word), hotword, content, flags=re.IGNORECASE)
        # content = re.sub(re.escape(word), replace.visual(word, replace.similar_0), content, flags=re.IGNORECASE)
        # hotword = replace.find_most_similiar(word, alternative_words)
        # if hotword:  # next(iter(hotword)
        #     content = re.sub(re.escape(word), next(iter(hotword)), content, flags=re.IGNORECASE)
        # else:
        #     content = re.sub(re.escape(word), replace.visual(word, replace.similar_0), content,
        #                      flags=re.IGNORECASE)  # 还可以再加参数限制改几个
    return sum_twords,sum_twords_invis,content
    # 如果不在明显的地方，则找其它类重点词中，拼写相近的替换，如果找不到，再用相似字符替换
    # hotword = replace.find_most_similiar(word, alternative_words)
    # if hotword: #next(iter(hotword)
    #     content = re.sub(re.escape(word), next(iter(hotword)), content, flags=re.IGNORECASE)
    # else:
    #     content = re.sub(re.escape(word), replace.visual(word, replace.similar_0), content, flags=re.IGNORECASE)  # 还可以再加参数限制改几个
    # return content
# (word_index, model, texts, embeddings,vocab,word, banner_head, alternative_words)
def invisible_server_repleace(word_index, model, texts, embeddings,vocab,word, banner_head, alternative_words):
    index_serv_start = 0
    tword = []
    if re.search('server:', banner_head, re.IGNORECASE) is None:
        # tword = replace.most_largest_cos_other(embeddings,vocab, word, alternative_words)
        tword = replace.replace_insert_sim(embeddings, vocab, word, alternative_words)
        # tword = replace.invisible_tramspose(word)
        # tword = replace.find_least_leven(word, alternative_words)
        #用tword替换word
        banner_head = re.sub(word, tword, banner_head, flags=re.IGNORECASE)
        # banner_head = re.sub(re.escape(word), replace.visual(word, replace.similar_0), banner_head, flags=re.IGNORECASE)
        # tword = replace.find_most_similiar(word, alternative_words)
        # if tword:
        # # print(word,tword)
        #     banner_head = re.sub(word, next(iter(tword)), banner_head, flags=re.IGNORECASE, count=1)
        # else:
        #     banner_head = re.sub(word, replace.visual(word, replace.similar_0), banner_head, flags=re.IGNORECASE, count=1)
    else:
        index_serv_start = re.search('server:', banner_head, re.IGNORECASE).span()[0]
        index_serv_end = banner_head.find('\r\n', index_serv_start, index_start)
        server = banner_head[index_serv_start:index_serv_end]
        # print(server)
        # server_c = re.sub(r'([\$\(\)])', r'\\\1',server)
        server_c = re.escape(server)
        if word in re.search(server_c, banner_head, flags=re.IGNORECASE).group(0):
            # tword = replace.most_largest_cos_other(embeddings, vocab, word, alternative_words)
            tword = replace.replace_insert_sim(embeddings, vocab, word, alternative_words)
            # tword = replace.invisible_tramspose(word)
            # tword = replace.find_least_leven(word, alternative_words)
            b = re.sub(word, tword, server, flags=re.IGNORECASE)
            # b = re.sub(re.escape(word), replace.visual(word, replace.similar_0), server,
            #                      flags=re.IGNORECASE)
            # tword = replace.find_most_similiar(word, alternative_words)
            # if tword:
            #     b = re.sub(word, next(iter(tword)), server, flags=re.IGNORECASE)
            # else:
            #     b = re.sub(word, replace.visual(word, replace.similar_0), server, flags=re.IGNORECASE)
            banner_head = re.sub(server_c, b, banner_head, flags=re.IGNORECASE)

    return tword,banner_head

def invisible_comment_replace(word_index, model, comment, embeddings,vocab,word, html_doc, alternative_words):
    # tword = replace.most_largest_cos_other(embeddings, vocab, word, alternative_words)
    tword = replace.replace_insert_sim(embeddings, vocab, word, alternative_words)
    # tword = replace.invisible_tramspose(word)
    # tword = replace.find_least_leven(word, alternative_words)
    # tword = replace.find_most_similiar(word, alternative_words)
    # if tword:
    #     comment = re.sub(word, next(iter(tword)), comment, flags=re.IGNORECASE)
    # else:
    #     comment = re.sub(word, replace.visual(word, replace.similar_0), comment, flags=re.IGNORECASE)
    comment = re.sub(word, tword, comment, flags=re.IGNORECASE)
    # comment = re.sub(word, replace.visual(word, replace.similar_0), comment, flags=re.IGNORECASE)
    return tword,comment

def get_same_element_index(ob_list, word):

    return [i+1 for (i, v) in enumerate(ob_list) if v == word]


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

    model.load_state_dict(torch.load(path.dt_cnn_modelpath,map_location=torch.device('cpu')))

    model = model.to(device)

    dataset = 'dataset-dt'
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
                        del s_note,e_note
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
                    #print(len(text))
                    texts = ' '.join(text)
                    #print(texts)
                    ori_text_list = utils.ori_preprocessing(input)
                    #print(len(ori_text_list))
                    # text = utils.extracTextFromHtml(content)
                    # tags, text = find_text_tag(content)
                    # texts = ' '.join(text)
                    # A = 3
                    # C = max(A, int(len(ori_text_list) * 0.2))
                    # C = min(A, len(ori_text_list))
                    # C = int(len(ori_text_list) * 0.2)
                    count = 0
                    H=[]
                    vis_words = {}
                    invis_words = {}
                    sum_vis = 0
                    sum_invis = 0
                    #succ = 0
                    #选取text即原始输入数据中在重点词表中的数据
                    for word in text:
                        if word in all_words:
                            H.append(word)
                    #print(H)
                    #print(".......")
                    #print(text)
                    # print(len(H))
                    # C = max(A, int(len(H) * 0.2))
                    # q = q+ len(H)
                    # print(str(len(H)) + '|' + str(C))
                    # if len(H1) > C:
                    #     # H = random.sample(H, 3)
                    #     l = {}
                    #     for a in H1:
                    #         l[a] = self_fs_words[a]
                    #     M = sorted(l.items(), key=lambda x: x[1], reverse=True)[:C]
                    #     H = []
                    #     for m in M:
                    #         H.append(m[0])
                    #     del M,l
                    # del H1
                    # else:
                    #     h = H
                    #原始数据分词后的单词数
                    text_len = len(ori_text_list)
                    pre_len = len(H)
                    num_changes = 0
                    #H = H[0:1]
                    for word in set(H):  # 是否在总重点词表中
                        #if (len(vis_words) + len(invis_words)) >=2:
                            #break
                        #重点词在text中的位置，0是1，某一个词在里面的位置
                        word_index_list = get_same_element_index(text, word)
                        for word_index in word_index_list:

                            if word_index < 100:
                        # content = region_repalce(word, content, alternative_words, embeddings = embeddings)(word_index, model, texts, embeddings,vocab, word, word_list)
                                #不可见区域
                                #if (len(vis_words)  + len(invis_words)) >= 2:
                                    #break
                                replace_word,banner_head = invisible_server_repleace(word_index, model, texts, embeddings,vocab,word, banner_head, alternative_words)
                                if len(replace_word) >= 1 :
                                    invis_words.update({text[word_index - 1]: replace_word})
                                    #sum_invis = sum_invis +1
                                #print("banner_head",banner_head)
                                if re.search('<!--[^>]*-->', html_doc):
                                    comments = re.findall('<!--[^>]*-->', html_doc)
                                    for comment in comments:
                                        #if (len(vis_words) + len(invis_words)) >= 2:
                                           # break
                                        replace_word1,comment_s = invisible_comment_replace(word_index, model, comment, embeddings,vocab,word, html_doc, alternative_words)
                                        html_doc = re.sub(re.escape(comment), comment_s, html_doc, flags=re.IGNORECASE)
                                        if len(replace_word1) >= 1:
                                            invis_words.update({text[word_index - 1]: replace_word1})
                                        del comment_s
                                    del comments
                                #if (len(vis_words) + len(invis_words)) >= 2:
                                    # break
                            #可见区域
                                replace_word_vis,replace_word_invis,html_doc = region_repalce(word_index, model, texts, embeddings,vocab,word, html_doc, alternative_words)
                                if len(replace_word_vis) >=1 and '220' not in banner_head:
                                    vis_words.update({text[word_index - 1]: replace_word_vis})
                                if len(replace_word_invis) >=1:
                                    invis_words.update({text[word_index - 1]: replace_word_invis})
                                #print("html",html_doc)
                                # count = len(H)
                        if len(vis_words) >= 1 or len(invis_words) >=1:
                            count = 1
                        for i in range(len(js_list)):
                            # print(js_list[i])
                            # content = re.sub(r'~~~', js_list[i].replace('\\', '/'), content, count=1)  # .replace('\\', '/')
                            html_doc = re.sub(r'~~~', js_list[i].replace('\\', '/'), html_doc, count=1)
                        content = banner_head + html_doc
                        #content1 = preprocess(content)
                        x_pred = predict(content)
                        x_tag = parse_tag(x_pred, config)
                        if x_tag[0] != dict['brand'].lower():
                            #succ=1
                            break

                        del word_index_list
                    # if count < C:
                    if count == 0:
                        t = 0

                        # 如果语句中不包含重点词表中的词，直接匹配字母与数字相结合的，以及不在语料库中的无意义的、不正常的词进行替换
                        specific_words = []
                        for word in text:
                            if re.search(name_pattern, word) or word not in word_set:
                                specific_words.append(word)

                                # count = count + 1
                                # if count == C:
                                #     break
                                #count = 1
                        pre_len = len(specific_words)
                        #specific_words = specific_words[0:1]
                        for word in set(specific_words):
                            #if (len(vis_words)  + len(invis_words)) >= 2:
                               #break
                            word_index_list = get_same_element_index(text, word)
                            #word_index_list = word_index_list[0:1]
                            for word_index in word_index_list:
                                if word_index < 100:
                            # content = region_repalce(word, content, alternative_words, embeddings = embeddings)
                                    #if (len(vis_words) + len(invis_words)) >= 2:
                                        #break
                                    replace_word,banner_head = invisible_server_repleace(word_index, model, texts, embeddings, vocab,
                                                                            word, banner_head, alternative_words)
                                    if len(replace_word) >= 1:
                                        invis_words.update({text[word_index - 1]: replace_word})
                                    #print("banner_head", banner_head)
                                    #print(banner_head)
                                    if re.search('<!--[^>]*-->', html_doc):
                                        comments = re.findall('<!--[^>]*-->', html_doc)
                                        for comment in comments:
                                            #if (len(vis_words) + len(invis_words)) >= 2:
                                               #break
                                            replace_word1,comment_s = invisible_comment_replace(word_index, model, comment, embeddings, vocab,
                                                                                  word, html_doc, alternative_words)
                                            if len(replace_word1) >= 1:
                                                invis_words.update({text[word_index - 1]: replace_word1})
                                                #sum_invis = sum_invis + 1
                                            print("______")
                                            print(comment_s)
                                            html_doc = re.sub(re.escape(comment), comment_s, html_doc, flags=re.IGNORECASE)
                                            del comment_s
                                        del comments
                                    #if (len(vis_words) + len(invis_words))>= 2:
                                        #break
                                    replace_word_vis,replace_word_invis,html_doc = region_repalce(word_index, model, texts, embeddings, vocab, word,
                                                              html_doc, alternative_words)
                                    if len(replace_word_vis) >= 1 and '220' not in banner_head:
                                         vis_words.update({text[word_index - 1]: replace_word_vis})
                                    if len(replace_word_invis) >=1:
                                        invis_words.update({text[word_index - 1]: replace_word_invis})
                                    #print("html_doc",html_doc)
                            del word_index_list
                            if len(vis_words) >= 1 or len(invis_words) >= 1:
                                count = 1
                            for i in range(len(js_list)):
                                # print(js_list[i])
                                # content = re.sub(r'~~~', js_list[i].replace('\\', '/'), content, count=1)  # .replace('\\', '/')
                                html_doc = re.sub(r'~~~', js_list[i].replace('\\', '/'), html_doc, count=1)
                            content = banner_head + html_doc
                            #content1 = preprocess(content)
                            x_pred = predict(content)
                            x_tag = parse_tag(x_pred, config)
                            if x_tag[0] != dict['brand'].lower():
                                #succ = 1
                                break
                        del specific_words

                    # if count < C:
                    if count == 0 or count == 1:

                        # 如果上两个都不满足，则使用get_hot_word.py为单个样本中的词衡量重要性，选取重要性前3的进行改动
                        # words = getwords.get_hot_word(texts, model, device, mode='word', power=C-count)
                        words = getwords.get_hot_word(texts, model, device, vocab,vocab_length=len(vocab),mode='word', power=5)
                        pre_len = len(words)
                        for word in words:
                            #if (len(vis_words) + len(invis_words)) >= 2:
                               # break
                            word_index_list = get_same_element_index(text, word)
                            for word_index in word_index_list:
                                if word_index < 100:
                            #content = region_repalce(word, content, alternative_words, embeddings = embeddings)
                                    #if (len(vis_words) + len(invis_words) )>= 2:
                                       # break
                                    replace_word,banner_head = invisible_server_repleace(word_index, model, texts, embeddings, vocab,
                                                                            word, banner_head, alternative_words)
                                    if len(replace_word) >= 1:
                                        invis_words.update({text[word_index - 1]: replace_word})
                                    if re.search('<!--[^>]*-->', html_doc):
                                        comments = re.findall('<!--[^>]*-->', html_doc)
                                        for comment in comments:
                                           # if (len(vis_words) + len(invis_words)) >= 2:
                                                # break
                                            replace_word1,comment_s = invisible_comment_replace(word_index, model, comment, embeddings, vocab,
                                                                                  word, html_doc, alternative_words)
                                            html_doc = re.sub(re.escape(comment), comment_s, html_doc, flags=re.IGNORECASE)
                                            if len(replace_word1) >= 1:
                                                invis_words.update({text[word_index - 1]: replace_word1})
                                                #sum_invis = sum_invis + 1
                                            del comment_s
                                        del comments
                                    #if (len(vis_words) + len(invis_words)) >= 2:
                                       # break
                            #vis可见区域，invis不可见区域
                                    replace_word_vis,replace_word_invis,html_doc = region_repalce(word_index, model, texts, embeddings, vocab, word,
                                                              html_doc, alternative_words)
                                    if len(replace_word_vis) >= 1 and '220' not in banner_head:
                                        vis_words.update({text[word_index - 1]: replace_word_vis})
                                    if len(replace_word_invis) >=1:
                                        invis_words.update({text[word_index - 1]: replace_word_invis})
                            del word_index_list
                            for i in range(len(js_list)):
                                # print(js_list[i])
                                # content = re.sub(r'~~~', js_list[i].replace('\\', '/'), content, count=1)  # .replace('\\', '/')
                                html_doc = re.sub(r'~~~', js_list[i].replace('\\', '/'), html_doc, count=1)
                            content = banner_head + html_doc
                            content1 = preprocess(content)
                            x_pred = predict(content)
                            x_tag = parse_tag(x_pred, config)
                            if x_tag[0] != dict['brand'].lower():
                                #succ = 1
                                break
                        del words
                    for i in range(len(js_list)):
                        # print(js_list[i])
                        # content = re.sub(r'~~~', js_list[i].replace('\\', '/'), content, count=1)  # .replace('\\', '/')
                        html_doc = re.sub(r'~~~', js_list[i].replace('\\', '/'), html_doc, count=1)
                    content = banner_head + html_doc
                    content1 = preprocess(content)
                    x_pred = predict(content)
                    x_tag = parse_tag(x_pred, config)

                    #content_token = utils.ori_preprocessing(content)
                    #for i in range (0,text_len):
                        #if ori_text_list[i] != content_token[i]:
                            #end_index = 0
                            #num_changes += 1

                    # print(content)
                    if dict['brand'].lower() != x_tag[0] :
                        succ = 1
                    else:
                        succ = 0
                    id = id + 1
                    standard_dict = {
                        'id':id,
                        'succ':succ,
                        'device_type': dict['device_type'],
                        'brand': dict['brand'],
                        'product': dict['product'],
                        #'ori_brand':x_ori[0],
                        'adv_brand': x_tag[0],
                        'visual': vis_words,
                        'invisual': invis_words,
                        'sum_visual': len(vis_words),
                        'sum_invisual': len(invis_words),
                        "sum_all":len(vis_words)+len(invis_words),
                        'banner':dict['banner'],
                        'adv_banner': content
                    }
                    json.dump(standard_dict, f2)
                    f2.write('\n')
                    del banner_head
                    del html_doc
                    del content
                    del dict
                    del standard_dict, index_start, js_list, css_script_note, note_start,texts,text,H,count,input,ori_text_list
        del all_words, other_words, other_words_set, self_words, self_words_set,alternative_words
    # print(q/3000)