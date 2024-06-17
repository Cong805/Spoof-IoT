#! /usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import path_config as path
import re
import json
import random
import utils
from bs4 import BeautifulSoup, NavigableString
from cspan2tspan import get_char2tok_span, char_span2tok_span
import char_replace
from model.evaluation import model
import LGS, LGS_P, LGS_D
import sys
'''
class Logger(object):
    def __init__(self,filename='lgs.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

        def write(self,message):
            self.terminal.write(message)
            self.log.write(message)
        #def flush(self):
           # pass
'''
import os
class Logger(object):
    def __init__(self, fileN="default.log",stream=sys.stdout):
        self.terminal = sys.stdout
        self.log = open(fileN, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def htmlTag_find_text_index(soup, html_tag, html_doc):
    region_index = []
    for head in soup.find_all(html_tag):
        text = head.get_text()
        # region_index.extend(text.split())
        text = " ".join(preprocess(text))
        if text:
            text = text.replace("(", r"\(").replace(")", r"\)").replace("*", "\*").replace(r".", "\.").replace(r"+",
                                                                                                               "\+").replace(
                r"?", "\?").replace(r"[", "\[").replace(r"]", "\]")

            search_res = re.search(r'{}'.format(text), r'{}'.format(html_doc), re.IGNORECASE)
            if search_res:
                char2tok_span = get_char2tok_span(html_doc)
                tok_span = char_span2tok_span(search_res.span(), char2tok_span)
                # print(tok_span,html_doc.split()[62])
                # exit()
                # start_index = search_res.span()[0]
                # end_index = search_res.span()[1]
                region_index.extend(list(range(tok_span[0], tok_span[1])))
    return region_index


def htmlTag_select_text_index(soup, html_tag, html_doc):
    region_index = []
    for head in soup.select(html_tag):
        text = head.get_text()
        # region_index.extend(text.split())
        text = " ".join(text.split())
        if text:
            text = text.replace("(", r"\(").replace(")", r"\)").replace("*", "\*").replace(r".", "\.").replace(r"+",
                                                                                                               "\+").replace(
                r"?", "\?").replace(r"[", "\[").replace(r"]", "\]")
            # print(text)
            search_res = re.search(r'{}'.format(text), r'{}'.format(html_doc), re.IGNORECASE)
            if search_res:
                char2tok_span = get_char2tok_span(html_doc)
                tok_span = char_span2tok_span(search_res.span(), char2tok_span)
                region_index.extend(list(range(tok_span[0], tok_span[1])))
                # start_index = search_res.span()[0]
                # end_index = search_res.span()[1]
                # region_index.extend(list(range(start_index, end_index + 1)))
    return region_index

#焦点呈现区域
def region_split_VFA(soup, banner_rm_html):
    region_VFA = []
    html_tag_list = ["h1", "h2", "h3", "h4", "h5", "h6", "b", "big", "em", "strong", "i", "u", "title"]
    for html_tag in html_tag_list:
        region_VFA.extend(htmlTag_find_text_index(soup, html_tag, banner_rm_html))
        # content = similar_replace(soup, title_items, title_text)
    return region_VFA

#
def region_split_VFSA(soup, banner_rm_html):
    region_VFSA = []
    html_tag_list0 = ["a"]
    html_tag_list1 = ["ul > li", "ol > li", "dl > dt", "dl > dd", "select > option"]
    for html_tag in html_tag_list0:
        region_VFSA.extend(htmlTag_find_text_index(soup, html_tag, banner_rm_html))
        # content = level2_replace(soup, a_items, a_text)
    for html_tag in html_tag_list1:
        region_VFSA.extend(htmlTag_select_text_index(soup, html_tag, banner_rm_html))
        # level2_replace(soup, option_items, option_text)
    return region_VFSA


def region_split_VUA(soup, banner_rm_html):
    region_VUA = []
    html_tag_list = ["table > caption", "table > tr > th", "table > tr > td"]
    for html_tag in html_tag_list:
        region_VUA.extend(htmlTag_select_text_index(soup, html_tag, banner_rm_html))
        # content = level3_replace(soup, ta_items, ta_text)
    return region_VUA


def region_split_other(soup, banner_rm_html):
    region_other = []
    if soup.find_all("body"):
        a = soup.body.descendants
        for i in a:
            if not isinstance(i, NavigableString):
                i_text = i.get_text()
                # region_other.extend(i_text.split())
                i_text = " ".join(preprocess(i_text))
                if i_text:
                    i_text = i_text.replace("(", r"\(").replace(")", r"\)").replace("*", "\*").replace(r".",
                                                                                                       "\.").replace(
                        r"+", "\+").replace(r"?", "\?").replace(r"[", "\[").replace(r"]", "\]")
                    search_res = re.search(r'{}'.format(i_text), r'{}'.format(banner_rm_html), re.IGNORECASE)
                    if search_res:
                        char2tok_span = get_char2tok_span(banner_rm_html)
                        tok_span = char_span2tok_span(search_res.span(), char2tok_span)
                        region_other.extend(list(range(tok_span[0], tok_span[1])))
                        # start_index = search_res.span()[0]
                        # end_index = search_res.span()[1]
                        # region_other.extend(list(range(start_index, end_index + 1)))
                    # content = level4_replace(soup, i, i_text)
    return region_other


def preprocess(input):
    # input -> token list
    text = utils.extracTextFromHtml(input)
    text = text.replace("/", " ").replace("=", " ").replace(":", "").replace("\"", "").replace("'", " ")
    token_list = text.split()
    return token_list


if __name__ == '__main__':

    sys.stdout = Logger("adv_d.log", sys.stdout)
    filelist = ["select_data.json"]

    for i, file in enumerate(filelist):
        data_path = path.raw_data_dir + '/' + file
        processed_data_path = path.processed_data_dir_b + '/' + "adv_d_1.json"

        cos_sim_words_dict = {}
        with open(path.sim_vocab_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                dic = json.loads(line)
                for word, word_list in dic.items():
                    cos_sim_words_dict[word] = word_list

        LGS_Attack = LGS_D.LGS_Attack(model, pop_size=20, max_iters=20)



        with open(data_path, 'r', encoding='utf-8') as f:
            with open(processed_data_path, 'w', encoding='utf-8') as f2:
                for line in f.readlines():
                    dict = json.loads(line)
                    #if int(dict["id"])<=2000:
                         #continue
                    if int(dict["id"])>20000:
                         break
                    #if int(dict["id"]) >= 1000:
                    #    break
                    id = dict["id"]

                    print("id", id)
                    input = dict['banner'].lower()
                    if dict['brand'] == 'AXIS' or 'FTP' in dict['banner']:
                        print("failed")
                        continue
                    html_flag = 0
                    index_start = 0
                    html_start_string_list = ['<!doctype html', '<?xml', '<html']
                    for html_start_string in html_start_string_list:
                        start_search_result = re.search(html_start_string, input, re.IGNORECASE)
                        if start_search_result:
                            index_start = start_search_result.span()[0]
                            html_flag = 1
                            break

                    if not html_flag:  # 非半结构化的banner数据
                        index_start = len(input)
                        banner_head = input
                        # html_doc = ''
                    else:  # 提取banner中HTML的内容
                        end_search_result = re.search('</html>', input, re.IGNORECASE)
                        if end_search_result:
                            index_end = end_search_result.span()[1]
                            banner_head = input[0:index_start]  # 不可见部分的一部分
                            # html_doc = input[index_start:index_end]
                        else:
                            banner_head = input[0:index_start]  # 不可见部分的一部分
                            # html_doc = input[index_start:]


                    token_list = preprocess(input)
                    banner_rm_html = ' '.join(token_list)
                    print("token_list", token_list)
                    pre_token_list = utils.ori_preprocessing(input)

                    perturbable_words_id = []
                    perturbable_words_substitution = {}
                    vis_words_id = []
                    vis_words = {}
                    sim_words_id = []
                    sim_words = {}
                    vis_num = 0
                    sem_num = 0
                    if not html_flag:
                        for index, token in enumerate(token_list):
                            if token in pre_token_list and len(token) >= 2 and token in cos_sim_words_dict.keys():
                                perturbable_words_id.append(index)
                                sim_words_id.append(index)
                                perturbable_words_substitution[index] = cos_sim_words_dict[token]
                                sim_words[token_list[index]] = cos_sim_words_dict[token]
                                sem_num = sem_num+1
                    else:
                        soup = BeautifulSoup(input, 'html.parser')
                        VFA_index = region_split_VFA(soup, banner_rm_html)
                        VFSA_index = region_split_VFSA(soup, banner_rm_html)
                        VUA_index = region_split_VUA(soup, banner_rm_html)
                        OTHER_index = region_split_other(soup, banner_rm_html)
                        HEAD_index = []
                        banner_head_token_list = preprocess(banner_head)
                        HEAD_index = list(range(len(banner_head_token_list)))

                        VFA_VFSA_index = list(set(VFA_index).union(set(VFSA_index)))

                        HEAD_VUA_OTHER_index = list(set(HEAD_index).union(set(VUA_index)).union(set(OTHER_index)))

                        # print("VFA_index",VFA_index)
                        # print("VFSA_index", VFSA_index)
                        # print("VUA_index", VUA_index)
                        # print("OTHER_index", OTHER_index)
                        # print("HEAD_index", HEAD_index)

                        # 匹配数字和字符结合的词
                        name_pattern = r"[0-9]+[a-zA-Z]+[0-9a-zA-Z]*|[a-zA-Z]+[0-9]+[0-9a-zA-Z]*"

                        temp = []

                        for index, token in enumerate(token_list):
                            if token in pre_token_list and len(token) >= 2:
                                if index in VFA_VFSA_index:  # 字符扰动
                                    temp.append(token)
                                    #vis_ori_token.append(token)
                                    perturbable_words_id.append(index)
                                    vis_words_id.append(index)
                                    perturbable_words_substitution[index] = char_replace.visual(token)
                                    vis_words[token_list[index]] = char_replace.visual(token)
                                    vis_num = vis_num+1
                                    #vis_adv_token = char_replace.visual(token)

                                elif index in HEAD_VUA_OTHER_index:  # 词扰动
                                    if token in cos_sim_words_dict.keys():
                                        temp.append(token)
                                        perturbable_words_id.append(index)
                                        perturbable_words_substitution[index] = cos_sim_words_dict[token]
                                        sim_words[token_list[index]] = cos_sim_words_dict[token]
                                        sem_num = sem_num +1
                                    # elif re.search(name_pattern, token):  # 词表不全，name_pattern做补充
                                    #     temp.append(token)
                                    #     perturbable_words_id.append(index)
                                    #     perturbable_words_substitution[index] = char_replace.visual(token)


                    print("perturbable_words", temp)
                    # exit()
                    if not perturbable_words_id:
                        continue
                    x_adv, query_count = LGS_Attack.attack(token_list, perturbable_words_id,
                                                           perturbable_words_substitution)
                    print("x_adv:",x_adv)

                    if x_adv is None:
                        print('failed')
                        # fail_list.append(test_idx[i])
                    else:
                        x_len = len(token_list)
                        banner_adv = input
                        pre_index = 0
                        num_changes = 0
                        ori_word = []
                        adv_word = []
                        for i in range(x_len):
                            if x_adv[i] != token_list[i]:
                                end_index = 0
                                num_changes += 1
                                ori_word.append(token_list[i])
                                adv_word.append(x_adv[i])
                                search_res = re.search(token_list[i], banner_adv, re.IGNORECASE)
                                if search_res:
                                    start_index = search_res.span()[0]
                                    end_index = search_res.span()[1]
                                banner_adv = banner_adv[:pre_index] + banner_adv[pre_index:].replace(token_list[i],
                                                                                                     x_adv[i], 1)
                                pre_index = end_index
                                # print('改了',token_list[i],x_adv[i])
                        #print("banner_adv",banner_adv)
                        token_adv = preprocess(banner_adv)
                        adv_result = LGS_Attack.prediction(token_adv)
                        print(adv_result)
                        modify_ratio = num_changes / x_len
                        dict['device_type'] = dict['device_type'].lower()
                        dict['brand'] = dict['brand'].lower()
                        dict['product'] =dict['product'].lower()
                        flag = [0,0,0]
                        if adv_result[0] == dict['device_type'] and adv_result[1] == dict['brand'] and adv_result[2] == dict['product']:
                            flag_dpv = 0
                        else:
                            flag_dpv = 1
                        #print('success! query_count', query_count)
                        if adv_result[0] == dict['device_type']:
                            flag[0] = 0
                        else:
                            flag[0] = 1
                        if adv_result[1] == dict['brand']:
                            flag[1] = 0
                        else:
                            flag[1] = 1
                        if adv_result[2] == dict['product']:
                            flag[2] = 0
                        else:
                            flag[2] = 1
                        print('......')
                        print(len(dict['banner']))
                        print(len(banner_adv))
                        standard_dict = {
                            "id": dict["id"],
                            "succ": flag,
                            "succ_device_type":flag[0],
                            "succ_brand" :flag[1],
                            "succ_product": flag[2],
                            "query_count": query_count,
                            "num_changes": num_changes,
                            "modify_ratio": modify_ratio,
                            'device_type': dict['device_type'],
                            'brand': dict['brand'],
                            'product': dict['product'],
                            'adv_device_type': adv_result[0],
                            'adv_brand': adv_result[1],
                            'adv_product': adv_result[2],
                            'visual' : vis_words,
                            'semantic': sim_words,
                            'sum_visual': vis_num,
                            'sum_semantic':sem_num,
                            "sum_all":vis_num+sem_num,
                            'banner':dict['banner'],
                            'adv_banner': banner_adv
                        }
                        json.dump(standard_dict, f2)
                        f2.write('\n')
                print('--------------------------')
