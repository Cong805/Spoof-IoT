import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import argparse
import numpy as np
import math
import os
import sys
import json
import io
import pickle as pkl
from tqdm import tqdm
import path_config as path
from bs4 import BeautifulSoup
import re
import difflib
import pickle

# def transform(word, similar):
#     #相似替换，将word中的第一个字母替换成视觉相似字母，返回替换后的词
#
#     # for c in word:
#     #     if c in similar:
#     #         word = re.sub(c, similar[c], word)
#     # return word
#     # s = np.random.randint(0, len(word))
#     s = np.random.randint(0, 1)
#     if word[s] in similar:
#         rletter = similar[word[s]]
#     else:
#         rletter = word[s]
#     cword = word[:s] + rletter + word[s+1:]
#     return cword
#
# #视觉相似字符对照表
# similar = {'-': '˗', '9': '৭', '8': 'Ȣ', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ч', '3': 'Ʒ', '2': 'ƻ', '1': 'l',
#                '0': 'O', "'": '`', 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ',
#                'i': 'і', 'j': 'ϳ', 'k': '𝒌', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ',
#                's': 'ѕ', 't': '𝚝', 'u': 'ս', 'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ'}
# #
# default_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
#
#
# default_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"

def preprocess(data, device):
    vocab = pickle.load(open('dataset-dt/data/vocab.pkl', 'rb'))
    pad_size = 100
    tokenizer = lambda x: x.split(' ')
    lin = data.strip()
    try:
        content, label = lin.split('\t')
    except ValueError:
        content = lin
        label = '-1'
    words_line = []
    token = tokenizer(content)
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    content = (words_line, float(label), seq_len)
    x = torch.LongTensor([content[0]]).to(device)
    y = torch.LongTensor([content[1]]).to(device)
    # pad前的长度(超过pad_size的设为pad_size)
    seq_len = torch.LongTensor([content[2]]).to(device)
    return (x, seq_len), y  # data, label = preprocess(data, device)
def get_same_element_index(ob_list, word):

    return [i+1 for (i, v) in enumerate(ob_list) if v == word]

def find_most_similiar(word, word_list):
    # 找到相似的词，是字符串拼写上的，eg.由v100r001找到v100r003
    a = difflib.get_close_matches(word, word_list, 1, cutoff=0.5)

    return a


def extracTextFromHtml(html):
    html_a = html
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except:
        return ""
    if str(type(soup.script)) != str("<type 'NoneType'>"):
        for scripts in soup.find_all('script'):
            if scripts.string:
                html_a = html_a.replace(str(scripts.string), '')
    if str(type(soup.style)) != str("<type 'NoneType'>"):
        for styles in soup.find_all('style'):
            if styles.string:
                html_a = html_a.replace(str(styles.string), '')
    try:
        soup = BeautifulSoup(html_a, 'ihtml.parser')
    except:
        soup = BeautifulSoup(html_a, 'lxml')
    text = soup.get_text()
    return text


UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

vocab = pkl.load(open(path.vocab_path, 'rb'))

vocab_length = len(vocab) - 2  # vocab的定义在utils中



def build_text(inputs, vocab):
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    pad_size = 100
    contents = []

    lin = inputs.strip()
    try:
        content, label = lin.split('\t')
    except ValueError:
        content = lin
        label = '-1'
    words_line = []
    token = tokenizer(content)
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    contents.append((words_line, float(label), seq_len))
    return words_line, torch.Tensor([float(label)]).long()  # [([...], 0), ([...], 1), ...]

def preprocess(data, device):
    #vocab = pickle.load(open('dataset/data/vocab.pkl', 'rb'))
    pad_size = 100
    tokenizer = lambda x: x.split(' ')
    lin = data.strip()
    try:
        content, label = lin.split('\t')
    except ValueError:
        content = lin
        label = '-1'
    words_line = []
    token = tokenizer(content)
    seq_len = len(token)
    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    # word to id
    for word in token:
        words_line.append(vocab.get(word, vocab.get(UNK)))
    content = (words_line, float(label), seq_len)
    x = torch.LongTensor([content[0]]).to(device)
    y = torch.LongTensor([content[1]]).to(device)

    # pad前的长度(超过pad_size的设为pad_size)
    seq_len = torch.LongTensor([content[2]]).to(device)
    return (x, seq_len), y  # data, label = preprocess(data, device)


def remove_duplicate(readPath, writePath):
    # 词表去重

    # readPath = 'o_hotwords.txt'
    # writePath = 'rd_hotwords.txt'
    lines_seen = set()
    outfiile = open(writePath, 'w', encoding='utf-8')
    f = open(readPath, 'r', encoding='utf-8')
    for line in f:
        if line not in lines_seen:
            outfiile.write(line)
            lines_seen.add(line)

def count_f(readPath,writePath):
    outfiile = open(writePath, 'w', encoding='utf-8')
    words_list = dictlist(readPath)
    d = {}
    for w in words_list:
        d[w] = d.get(w, 0) + 1
    # print(d)
    # M= list(d.items())
    # print(M)
    # M.sort(key=lambda x: x[1], reverse=True)
    M = sorted(d.items(), key=lambda x: x[1], reverse=True)
    # for m in M:
    #     wd = {m[0]: m[1]}
    #     outfiile.write(str(wd))
    #     outfiile.write('\n')
    #     outfiile.write(m[0])
    #     outfiile.write('\n')
    outfiile.write(json.dumps(dict(M)))


def build_dataset(config, data_path, vocab):
    tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level

    def load_dataset(path, pad_size, vocab):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                try:
                    content, label = lin.split('\t')
                except ValueError:
                    content = lin
                    label = '-1'
                words_line = []
                token = tokenizer(content)
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))
                contents.append((words_line, float(label), seq_len))
                # print(contents)
        return contents  # [([...], 0), ([...], 1), ...]

    # train = load_dataset(config.val_path, config.pad_size, vocab)
    train = load_dataset(data_path, config.pad_size, vocab)
    return train


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.batch_size != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def ori_preprocessing(text):
    # 从网页中提取文本信息, 非网页文本保持不变
    text = extracTextFromHtml(text)
    text = ' '.join(text.split())
    # print(text)
    # 去除时间、ip地址等信息
    time_pattern1 = "(0\d{1}|1\d{1}|2[0-3]):([0-5]\d{1})"
    time_pattern2 = "(0\d{1}|1\d{1}|2[0-3]):[0-5]\d{1}:([0-5]\d{0,1})"
    ip_pattern = " ((?:(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))\.){3}(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d))))"
    mac_pattern = "([A-Fa-f0-9]{2}:){5}[A-Fa-f0-9]{2}"
    # Mon, 27 Mar 2006
    date_pattern1 = "(Mon|mon|Tues|tues|Tue|tue|Wed|wed|Thur|thur|Thu|thu|Fri|fri|Sat|sat|Sun|sun),\s+(\d|\d\d)\s+(Jan|jan|Feb|feb|Mar|mar|Apr|apr|May|may|Jun|jun|Jul|jul|Aug|aug|Sept|sept|Oct|oct|Nov|nov|Dec|dec)\s+\d\d\d\d"
    # Thu May 18 23:42:35 2000
    date_pattern2 = "(Mon|mon|Tues|tues|Tue|tue|Wed|wed|Thur|thur|Thu|thu|Fri|fri|Sat|sat|Sun|sun)\s+(Jan|jan|Feb|feb|Mar|mar|Apr|apr|May|may|Jun|jun|Jul|jul|Aug|aug|Sept|sept|Oct|oct|Nov|nov|Dec|dec)\s+(\d|\d\d)\s+\d\d:\d\d:\d\d\s+\d\d\d\d"
    # 单独的数字
    # number = '(^\d+\s+)|(\s\d+\s)|(,\d+\s)|(\s\d+$)'
    number = r'(\b\d+\b)'
    # 单独的字母
    schar = '(^[A-Za-z]\s+)|(\\b[A-Za-z]\s)|\s+[A-Za-z]$'

    p_date1 = re.compile(date_pattern1)
    p_date2 = re.compile(date_pattern2)
    p_mac = re.compile(mac_pattern)
    p_ip = re.compile(ip_pattern)
    p_time1 = re.compile(time_pattern1)
    p_time2 = re.compile(time_pattern2)
    p_number = re.compile(number)
    p_schar = re.compile(schar)

    text = p_date1.sub(" ", text)
    text = p_date2.sub(" ", text)
    text = p_mac.sub(" ", text)
    text = p_ip.sub(" ", text)
    text = p_time2.sub(" ", text)
    text = p_time1.sub(" ", text)
    text = p_number.sub("", text)
    text = p_schar.sub("", text)

    text = text.replace(r'\r', ' ').replace(r'\n', ' ').replace(r'\t', ' ')  # 去除\r, \n, \t
    text = re.sub(u"([^\u0030 -\u0039\u0041 -\u005a\u0061-\u007a])", '', text)  # 提取英文字符和数字
    text = ' '.join(re.split('/|=|:', text))
    # print(text)

    tokens = [word for word in word_tokenize(text)]  # 分词
    tokens = [word.lower() for word in tokens]  # 大小写转换，统一为小写
    tokens = [word.strip('-') for word in tokens]  # 去除单词首尾特殊符号
    tokens = [word.strip('.') for word in tokens]
    stop = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop]  # 去停词
    characters = ['', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', "'", "''", '``', '..',
                  '!', '*', '@', '#', '$', '%', '-', '...', '|', '=', '+', '//', "'s", "n't"]
    tokens = [word for word in tokens if word not in characters]  # 去特殊字符
    # print(tokens)
    return tokens

def preprocessing(text):
    text = ' '.join(text.split())
    # js文件名
    js = r'\b\w+\.js\b'
    # 域名，带有域名的网址
    realm = '([a-zA-Z0-9][-a-zA-Z0-9]{1,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{1,62})+\.?)' \
            '|((https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|])'

    request_parameters = r'((login|access|user-|transfer-|report-|accept-?|content-?|www-?)\S*)' \
                         '|\\b(allow(ed)?|application|gui|date|please|enter|expires|via|warning|vary|max-age|apply|trying|try|set-cookie|webserver|continue' \
                         '|loading|request(ed)?|location|unsafe-eval|located|error|timeout|chunked|private|close|pragma|found|find|keep-alive|reserved|later|forbidden' \
                         '|reports?|disabled|enable|session|link|log|connection|seconds|debut|click|help|service|no-cache|no-store|cache-control|gmt|url)\\b'

    header = r'(utc-?\S*)|devices?|accounts?|required?|support(ed|s)?|networks?' \
             '|\\b(control|time|configurator|welcome|static|rights|resource|unknown|type' \
             '|configuration|ffff|serial|os|systems?|status|series|us-ascii|resolution|ready' \
             '|realm|wireless|secure|live|website|web|((un)?authorized)|authorization|auth|xml|text|python|php)\\b'

    htmls = '\\b(html?|user(\s)?name|administrator|default|password|sign|color|images?|image|img|space|method' \
            '|radio|path|select|right|index|false|basic|correct|form|li|copyright|page|((un)?available)|data|view' \
            '|admin|administrator|using|use|nbsp|list|banner|inc|file|meta|size|exist|js|script|javascript|charset|utf-8)\\b'

    protocol = r'\b(version|ftpd?|cseq|httpd?|uhttpd?|rtsp|ok|server|(ipv\d+))\b'

    other = r'\b((\d+mbps)|uk|usa|iso-8859-1|mes|welcome|asia|africa|canada|indian|deutsch|america|english|en|ltd|gzip|co|ch)\b|(\+)|(\')'
    # 日期形如: Thu, 27 Mar 2006 23:42:35
    date_pattern3 = "(Mon|mon|Tues|tues|Tue|tue|Wed|wed|Thur|thur|Thu|thu|Fri|fri|Sat|sat|Sun|sun),\s(\d|\d\d)\s(Jan|jan|Feb|feb|Mar|mar|Apr|apr|May|may|Jun|jun|Jul|jul|Aug|aug|Sept|sept|sep|Sep|Oct|opt|Nov|nov|Dec|dec)\s\d\d\d\d\s\d\d:\d\d:\d\d"
    # 日期形如: 01-jan-1970 或 01 jan 1970
    date_pattern4 = '(0[1-9]|[1-2][0-9]|30|31)-?(Jan|jan|Feb|feb|Mar|mar|Apr|apr|May|may|Jun|jun|Jul|jul|Aug|aug|Sept|sept|sep|Sep|Oct|opt|Nov|nov|Dec|dec)-?[0-9]{4}'
    # 星期
    date_pattern5 = r'\b(mon|tues|tue|wed|thur|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|sept|sep|oct|nov|dec)\b'
    # 年份-年份
    date_pattern6 = r'\b([0-9]{4}-[0-9]{4})\b'
    # 指令码
    instruction = r'\b(0[xX][0-9a-fA-F]{8})\b'
    # IP地址
    ip = r'\b((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    # 单独的小数
    onep = '(^\d+\.\d+\s+)|(\s\d+\.\d+\s)|(,\d+\.\d+\s)'
    # 单独的数字
    # number = '(^\d+\s+)|(\s\d+\s)|(,\d+\s)|(\s\d+$)'
    number = r'(\b\d+\b)'
    # 单独的字母
    schar = '(^[A-Za-z]\s+)|(\\b[A-Za-z]\s)|\s+[A-Za-z]$'

    p_js = re.compile(js)
    p_realm = re.compile(realm)
    p_request_parameters = re.compile(request_parameters)
    p_header = re.compile(header)
    p_html = re.compile(htmls)
    p_protocol = re.compile(protocol)
    p_other = re.compile(other)
    p_date_pattern3 = re.compile(date_pattern3)
    p_date_pattern4 = re.compile(date_pattern4)
    p_date_pattern5 = re.compile(date_pattern5)
    p_date_pattern6 = re.compile(date_pattern6)
    p_instruction = re.compile(instruction)
    p_ip = re.compile(ip)
    p_onep = re.compile(onep)
    p_number = re.compile(number)
    p_schar = re.compile(schar)
    # p_middle = re.compile(middle)
    # p_tail = re.compile(tail)

    text = p_js.sub("", text)
    text = p_realm.sub("", text)
    text = p_request_parameters.sub("", text)
    text = p_header.sub("", text)
    text = p_html.sub("", text)
    text = p_protocol.sub("", text)
    text = p_other.sub("", text)
    text = p_date_pattern3.sub("", text)
    text = p_date_pattern4.sub("", text)
    text = p_date_pattern5.sub("", text)
    text = p_date_pattern6.sub("", text)
    text = p_instruction.sub("", text)
    text = p_ip.sub("", text)
    text = p_onep.sub(" ", text)
    text = p_number.sub("", text)
    text = p_schar.sub("", text)
    # text = p_middle.sub(" ",text)
    # text = p_tail.sub("",text)
    # print(text)
    text = text.replace(r'\r', ' ').replace(r'\n', ' ').replace(r'\t', ' ')  # 去除\r, \n, \t
    text = re.sub(u"([^\u0030 -\u0039\u0041 -\u005a\u0061-\u007a])", '', text)  # 提取英文字符和数字
    text = ' '.join(re.split('/|=|:', text))

    tokens = [word for word in word_tokenize(text)]  # 分词
    tokens = [word.lower() for word in tokens]  # 大小写转换，统一为小写
    tokens = [word.strip('-') for word in tokens]  # 去除单词首尾特殊符号
    tokens = [word.strip('.') for word in tokens]
    stop = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop]  # 去停词
    characters = ['', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', "'", "''", '``', '..',
                  '!', '*', '@', '#', '$', '%', '-', '...', '|', '=', '+', '//', "'s", "n't", '<', '>']
    tokens = [word for word in tokens if word not in characters]  # 去特殊字符
    # print(tokens)
    return tokens


#从头处理未处理过的数据，加入模型预处理数据时的正则
def raw_preprocessing(text):
    # 从网页中提取文本信息, 非网页文本保持不变
    text = extracTextFromHtml(text)
    text = ' '.join(text.split())
    # print(text)
    # 去除时间、ip地址等信息

    time_pattern1 = "(0\d{1}|1\d{1}|2[0-3]):([0-5]\d{1})"
    time_pattern2 = "(0\d{1}|1\d{1}|2[0-3]):[0-5]\d{1}:([0-5]\d{0,1})"
    ip_pattern = " ((?:(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))\.){3}(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d))))"
    mac_pattern = "([A-Fa-f0-9]{2}:){5}[A-Fa-f0-9]{2}"
    # Mon, 27 Mar 2006
    date_pattern1 = "(Mon|mon|Tues|tues|Tue|tue|Wed|wed|Thur|thur|Thu|thu|Fri|fri|Sat|sat|Sun|sun),\s(\d|\d\d)\s(Jan|jan|Feb|feb|Mar|mar|Apr|apr|May|may|Jun|jun|Jul|jul|Aug|aug|Sept|sept|sep|Sep|Oct|opt|Nov|nov|Dec|dec)\s\d\d\d\d"
    # Thu May 18 23:42:35 2000
    date_pattern2 = "(Mon|mon|Tues|tues|Tue|tue|Wed|wed|Thur|thur|Thu|thu|Fri|fri|Sat|sat|Sun|sun)\s(Jan|jan|Feb|feb|Mar|mar|Apr|apr|May|may|Jun|jun|Jul|jul|Aug|aug|Sept|sept|sep|Sep|Oct|opt|Nov|nov|Dec|dec)\s(\d|\d\d)\s\d\d:\d\d:\d\d\s\d\d\d\d"
    # date_pattern3 = "(Mon|Tues|Tue|Wed|Thur|Thu|Fri|Sat|Sun),\s(\d|\d\d)\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)\s\d\d\d\d\s\d\d:\d\d:\d\d"
    # js文件名
    js = r'\b\w+\.js\b'
    # 域名，带有域名的网址
    realm = '([a-zA-Z0-9][-a-zA-Z0-9]{1,62}(\.[a-zA-Z0-9][-a-zA-Z0-9]{1,62})+\.?)' \
            '|((https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|])'

    request_parameters = r'((login|access|user-|(T|t)ransfer-|report-|accept-?|(C|c)ontent-?|www-?)\S*)' \
                         '|\\b(allow(ed)?|application|gui|date|please|enter|(E|e)xpires|via|warning|vary|max-age|apply|trying|try|set-cookie|webserver|continue' \
                         '|loading|request(ed)?|location|unsafe-eval|located|error|timeout|chunked|private|close|(P|p)ragma|found|find|keep-alive|reserved|later|forbidden' \
                         '|reports?|disabled|enable|session|link|log|connection|seconds|debut|click|help|service|no-cache|no-store|(C|c)ache-(C|c)ontrol|url)\\b'

    header = r'(utc-?\S*)|devices?|accounts?|(R|r)equired?|support(ed|s)?|networks?' \
             '|\\b(control|time|configurator|welcome|static|rights|resource|unknown|type' \
             '|configuration|ffff|serial|os|systems?|status|series|us-ascii|resolution|ready' \
             '|realm|wireless|secure|live|website|web|(((U|u)n)?authorized)|(A|a)uthorization|auth|xml|text|python|php)\\b'

    htmls = '\\b(html?|user(\s)?name|administrator|default|password|sign|color|images?|image|img|space|method' \
            '|radio|path|select|right|index|false|basic|correct|form|li|copyright|page|((un)?available)|data|view' \
            '|admin|administrator|using|use|nbsp|list|banner|inc|file|meta|size|exist|js|script|javascript|charset|(utf|UTF)-8)\\b'

    protocol = r'\b(version|ftpd?|cseq|httpd?|HTTP|uhttpd?|rtsp|ok|OK|(S|s)erver|(ipv\d+))\b'

    other = r'\b((\d+mbps)|uk|usa|iso-8859-1|mes|welcome|asia|africa|canada|indian|deutsch|america|english|en|ltd|gzip|co|ch)\b|(\+)|(\')'
    # 日期形如: Thu, 27 Mar 2006 23:42:35
    date_pattern3 = "(Mon|mon|Tues|tues|Tue|tue|Wed|wed|Thur|thur|Thu|thu|Fri|fri|Sat|sat|Sun|sun),\s(\d|\d\d)\s(Jan|jan|Feb|feb|Mar|mar|Apr|apr|May|may|Jun|jun|Jul|jul|Aug|aug|Sept|sept|sep|Sep|Oct|opt|Nov|nov|Dec|dec)\s\d\d\d\d\s\d\d:\d\d:\d\d"
    # 日期形如: 01-jan-1970 或 01 jan 1970
    date_pattern4 = '(0[1-9]|[1-2][0-9]|30|31)-?(Jan|jan|Feb|feb|Mar|mar|Apr|apr|May|may|Jun|jun|Jul|jul|Aug|aug|Sept|sept|sep|Sep|Oct|opt|Nov|nov|Dec|dec)-?[0-9]{4}'
    # 星期
    date_pattern5 = r'\b(Mon|mon|Tues|tues|Tue|tue|Wed|wed|Thur|thur|Thu|thu|Fri|fri|Sat|sat|Sun|sun|Jan|jan|Feb|feb|Mar|mar|Apr|apr|May|may|Jun|jun|Jul|jul|Aug|aug|Sept|sept|sep|Sep|Oct|opt|Nov|nov|Dec|dec)\b'
    # 年份-年份
    date_pattern6 = r'\b([0-9]{4}-[0-9]{4})\b'
    # 指令码
    instruction = r'\b(0[xX][0-9a-fA-F]{8})\b'
    # IP地址
    ip = r'\b((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    # 单独的小数
    onep = '(^\d+\.\d+\s+)|(\s\d+\.\d+\s)|(,\d+\.\d+\s)'
    # 单独的数字
    # number = '(^\d+\s+)|(\s\d+\s)|(,\d+\s)|(\s\d+$)'
    number = r'(\b\d+\b)'
    # 单独的字母
    schar = '(^[A-Za-z]\s+)|(\\b[A-Za-z]\s)|\s+[A-Za-z]$'
    p_date1 = re.compile(date_pattern1)
    p_date2 = re.compile(date_pattern2)
    p_date3 = re.compile(date_pattern3)
    p_mac = re.compile(mac_pattern)
    p_ip = re.compile(ip_pattern)
    p_time1 = re.compile(time_pattern1)
    p_time2 = re.compile(time_pattern2)

    p_js = re.compile(js)
    p_realm = re.compile(realm)
    p_request_parameters = re.compile(request_parameters)
    p_header = re.compile(header)
    p_html = re.compile(htmls)
    p_protocol = re.compile(protocol)
    p_other = re.compile(other)
    p_date_pattern3 = re.compile(date_pattern3)
    p_date_pattern4 = re.compile(date_pattern4)
    p_date_pattern5 = re.compile(date_pattern5)
    p_date_pattern6 = re.compile(date_pattern6)
    p_instruction = re.compile(instruction)
    p_ip2 = re.compile(ip)
    p_onep = re.compile(onep)
    p_number = re.compile(number)
    p_schar = re.compile(schar)


    text = p_date1.sub(" ", text)
    text = p_date2.sub(" ", text)
    text = p_date3.sub(" ", text)
    text = p_mac.sub(" ", text)
    text = p_ip.sub(" ", text)
    text = p_time2.sub(" ", text)
    text = p_time1.sub(" ", text)

    text = p_js.sub("", text)
    text = p_realm.sub("", text)
    text = p_request_parameters.sub("", text)
    text = p_header.sub("", text)
    text = p_html.sub("", text)
    text = p_protocol.sub("", text)
    text = p_other.sub("", text)
    text = p_date_pattern3.sub("", text)
    text = p_date_pattern4.sub("", text)
    text = p_date_pattern5.sub("", text)
    text = p_date_pattern6.sub("", text)
    text = p_instruction.sub("", text)
    text = p_ip2.sub("", text)
    text = p_onep.sub(" ", text)
    text = p_number.sub("", text)
    text = p_schar.sub("", text)

    text = text.replace(r'\r', ' ').replace(r'\n', ' ').replace(r'\t', ' ')  # 去除\r, \n, \t
    text = re.sub(u"([^\u0030 -\u0039\u0041 -\u005a\u0061-\u007a])", '', text)  # 提取英文字符和数字
    text = ' '.join(re.split('/|=|:', text))
    # print(text)

    tokens = [word for word in word_tokenize(text)]  # 分词
    tokens = [word.lower() for word in tokens]  # 大小写转换，统一为小写
    tokens = [word.strip('-') for word in tokens]  # 去除单词首尾特殊符号
    tokens = [word.strip('.') for word in tokens]
    stop = stopwords.words('english')
    tokens = [word for word in tokens if word not in stop]  # 去停词
    characters = ['', ',', '.', ':', ';', '?', '(', ')', '[', ']', '&', "'", "''", '``', '..',
                  '!', '*', '@', '#', '$', '%', '-', '...', '|', '=', '+', '//', "'s", "n't"]
    tokens = [word for word in tokens if word not in characters]  # 去特殊字符

    return tokens


def dictlist(filepath):
    # 将重点词表的txt转成列表型式
    dicts = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return dicts


def get_item(list):
    # 从count_useless的返回结果中取出相应词

    result = []
    for tup in list:
        result.append(tup[0])
    return result


def count_useless(path):
    # 计算词频最高的单词
    # 返回的是[（词：词频），（词：词频）]

    from collections import Counter
    count = []
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().split()
        for word in text:
            if len(word) == 1:
                text.remove(word)
        # print(text)
    return Counter(text).most_common(500)


def remove_useless(file, ge):
    # 从某文件中去除列表ge中的词，目前没有在用，之前处理求得的词表时用了

    with open(file, 'r', encoding='utf-8')as f:
        lines = f.readlines()
    # ge = ['could','date','reset','need','ready',"auth","user","etc", "able",'track','hard', 'head', 'bad', 'request', 'blog', 'transport', 'language', 'adres', 'app', 'trust', 'css', 'clean', 'strip', 'return', 'complete', '\d+kbps', 'value', 'still', 'minute', 'colour', 'email', 'fail', 'yes', 'wrong', 'save', 'please', 'block','see','hour','nonce','number','speed','get','connect','log','command','privacy','policy','main','busy',]
    new = ''
    for line in lines:
        mode = True
        for i in ge:
            if i in line:
                mode = False
                break
        if mode:
            new += line
    with open(file, 'w')as f:
        f.write(new)


# 以下三个函数是求余弦相似度的目前用不到，先忽略
def get_att_dis(target, behaviored):
    attention_distribution = []

    for i in range(behaviored.size(0)):
        attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
        attention_distribution.append(attention_score)
    attention_distribution = torch.Tensor(attention_distribution)

    return attention_distribution / torch.sum(attention_distribution, 0)  # 标准化


def find_syns(embeddings, wordid):
    a = torch.FloatTensor(embeddings[(wordid)]).unsqueeze(dim=0)

    b = torch.FloatTensor(np.delete(embeddings, embeddings[(wordid)], axis=0))

    similarity = get_att_dis(target=a, behaviored=b)
    return -similarity


def most_syn(embeddings, wordid):
    sort_syn, ind = torch.sort(find_syns(embeddings, wordid))

    # print(list(utils.vocab.keys())[list(utils.vocab.values()).index(wordid)])
    return torch.FloatTensor(embeddings[(ind[0])]).unsqueeze(dim=0)
