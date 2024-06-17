import torch
import numpy as np
import path_config as path

import re
import os
from nltk.corpus import stopwords
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from importlib import import_module
import json
import utils
import replace
from models import TextRNN_Att, TextRCNN, TextCNN
import get_hot_word as getwords
#基于规则黑盒最最最最最最最初版




#从头处理未处理过的数据，加入模型预处理数据时的正则
def raw_preprocessing(text):
    # 从网页中提取文本信息, 非网页文本保持不变
    text = utils.extracTextFromHtml(text)
    text = ' '.join(text.split())
    # print(text)
    # 去除时间、ip地址等信息

    time_pattern1 = "(0\d{1}|1\d{1}|2[0-3]):([0-5]\d{1})"
    time_pattern2 = "(0\d{1}|1\d{1}|2[0-3]):[0-5]\d{1}:([0-5]\d{0,1})"
    ip_pattern = " ((?:(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))\.){3}(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d))))"
    mac_pattern = "([A-Fa-f0-9]{2}:){5}[A-Fa-f0-9]{2}"
    # Mon, 27 Mar 2006
    date_pattern1 = "(Mon|Tues|Tue|Wed|Thur|Thu|Fri|Sat|Sun),\s(\d|\d\d)\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)\s\d\d\d\d"
    # Thu May 18 23:42:35 2000
    date_pattern2 = "(Mon|Tues|Tue|Wed|Thur|Thu|Fri|Sat|Sun)\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)\s(\d|\d\d)\s\d\d:\d\d:\d\d\s\d\d\d\d"
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
    date_pattern3 = "(Mon|Tues|Tue|Wed|Thur|Thu|Fri|Sat|Sun),\s(\d|\d\d)\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sept|Oct|Nov|Dec)\s\d\d\d\d\s\d\d:\d\d:\d\d"
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
    #将重点词表的txt转成列表型式
    dicts = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return dicts



if __name__ == '__main__':
    #生成对抗样本，找到符合要求的词后，直接在未经处理的banner上改动

    torch.manual_seed(8)
    torch.cuda.manual_seed(8)

    embedding = 'embedding_banner.npz'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'TextRNN_Att'

    dataset = 'dataset-dt'
    x = import_module('models.' + model_name)
    config = x.Config(path.dataset, embedding)
    model = x.Model(config)

    model.load_state_dict(torch.load(config.save_path))

    model = model.to(device)

    # dataset = 'dataset-dt'
    info = np.load(dataset + '/data/' + embedding, allow_pickle=True)
    embeddings = info['embeddings'][()]

    vocab = utils.vocab


    #brown语料库，其中是常用的有意义的英文词汇
    word_list = brown.words()
    word_set = set(word_list)
    #正则，匹配数字和字符结合的词
    name_pattern = r"[0-9]+[a-zA-Z-]+[0-9.a-zA-Z]*|[a-zA-Z]+[0-9-]+[0-9.a-zA-Z]*"
    server_partten = r"(?<=(s|S)erver: )[^.\s]*"
    realm_partten = r"(?<=realm=)[^.\s]*|(?<=Realm=)[^.\s]*"
    charset_partten = r"(?<=charset=)[^.\s]*|(?<=encoding=)[^.\s]*"
    title_partten = r"<title.*?>(.*?)</title>"
    useless_partten = r"debut|Debut|D-LINK|d-link|WIRELESS|wireless|opt1|ats|Mrvl|apache|httpd|code6|epson|EPSON|mfp|ex4300-48p|b2375ԁnf|8570dn|Printer with Embedded Web Server|Xerox|xerox|HP HTTP Server|m-11|layer3|b312|c100|s200" \
                      r"|D-LINK SYSTEMS, INC. \| WIRELESS ROUTER"

    filelist = os.listdir(path.raw_data_dir)

    for i, file in enumerate(filelist):
        data_path = path.raw_data_dir + '/' + file
        words_path = path.hotwords_dir + '/' + file
        processed_data_path = path.processed_data_dir + '/' + file
        print('Processing ', data_path)
        other_words = []  # 其它类的 hot words
        all_words = []  # 总重点词表

        # 加载其它类的 hot words
        remaining_filelist = filelist[0:i] + filelist[i + 1:]
        for remaining_file in remaining_filelist:
            other_words.extend(utils.dictlist(path.hotwords_dir + '/' + remaining_file))
        other_words_set = set(other_words)
        # 加载本类的 hot words
        self_words = utils.dictlist(words_path)
        self_words_set = set(self_words)
        all_words.extend(list(other_words_set))
        all_words.extend(list(self_words_set))

        with open(data_path, 'r', encoding='utf-8') as f:
            with open(processed_data_path, 'w', encoding='utf-8') as f2:
                for line in f.readlines():
                    dict = json.loads(line)
                    content = dict['banner']

                    js_list = re.findall('<script.*?>[\s\S]*?</script>', content, flags=re.IGNORECASE)
                    charset_list = re.findall(charset_partten, content, flags=re.IGNORECASE)


                    for i in range(len(charset_list)):
                        content = re.sub(re.escape(charset_list[i]), r'%%%', content)

                    for i in range(len(js_list)):
                        content = re.sub(re.escape(js_list[i]), r'~~~', content)#所有的js代码要保持不变

                    text = raw_preprocessing(content)
                    texts = ' '.join(text)
                    content_words = utils.extracTextFromHtml(content)
                    content_words = ' '.join(content_words.split())

                    # content = ' '.join(text)

                    for word in text:
                        if word in all_words and len(word) > 1:
                            # 是否在总重点词表中

                            content = re.sub(re.escape(word), replace.visual(word, replace.similar_0, all=True), content,
                                             flags=re.IGNORECASE)  # 还可以再加参数限制改几个

                        # 如果语句中不包含重点词表中的词，直接匹配字母与数字相结合的，以及不在语料库中的无意义的、不正常的词进行替换
                        elif re.search(name_pattern, word) and len(word) > 1:

                            content = re.sub(re.escape(word), replace.visual(word, replace.similar_0, all=True), content,
                                             flags=re.IGNORECASE)
                        elif word not in word_set and len(word) > 1:
                            content = re.sub(re.escape(word), replace.visual(word, replace.similar_0, all=True), content,
                                             flags=re.IGNORECASE)

                    if re.findall(server_partten, content):
                        # word = re.search(server_partten,content).group()
                        for word in re.findall(server_partten, content):
                            if len(word) > 1:
                                content = re.sub(re.escape(word), replace.visual(word, replace.similar_0, all=True), content,
                                                 flags=re.IGNORECASE)

                    if re.findall(realm_partten, content):
                        # word = re.search(server_partten,content).group()
                        for word in re.findall(realm_partten, content):
                            if len(word) > 1:
                                content = re.sub(re.escape(word), replace.visual(word, replace.similar_0, all=True), content,
                                                 flags=re.IGNORECASE)

                    if re.findall(r"[0-9]+[a-zA-Z-]+[0-9a-zA-Z]*|[a-zA-Z-]+[0-9]+[0-9a-zA-Z]*", content_words):
                        for word in re.findall(r"[0-9]+[a-zA-Z-]+[0-9a-zA-Z]*|[a-zA-Z-]+[0-9]+[0-9a-zA-Z]*", content_words):
                            if len(word) > 1:
                                content = re.sub(re.escape(word), replace.visual(word, replace.similar_0, all=True), content,
                                                 flags=re.IGNORECASE)

                    for i in range(len(charset_list)):
                        content = re.sub(r'%%%', charset_list[i], content, count=1)

                    for i in range(len(js_list)):
                        # print(js_list[i])
                        content = re.sub(r'~~~', js_list[i], content, count=1)

                    if re.search(r'<!--[\s\S]*?-->', content):
                        content = re.sub(r'<!--[\s\S]*?-->', '', content)

                    if re.findall(r"<title.*?>[\s\S]*?</title>", content):
                        title_o = re.findall(r"<title.*?>[\s\S]*?</title>", content)[0]
                        title = ''.join(title_o)

                        temp = re.findall(title_partten, content)
                        if temp:
                            temp_list = re.split(r'[#$@/\\()<>{}[\] &]', temp[0])
                            for word in temp_list:
                                if re.search(r"\W", word) == None:
                                    title = re.sub(re.escape(word), replace.visual(word, replace.similar_0, all=True), title,
                                                       flags=re.IGNORECASE)
                                    # print(title)
                            content = re.sub(re.escape(title_o), title, content, flags=re.IGNORECASE)

                    useless = re.findall(useless_partten, content)
                    if useless:
                        for word in useless:
                            content = re.sub(word, replace.visual(re.escape(word), replace.similar_0), content,
                                             flags=re.IGNORECASE)

                    standard_dict = {
                        'device_type': dict['device_type'],
                        'brand': dict['brand'],
                        'product': dict['product'],
                        'banner': content
                    }
                    json.dump(standard_dict, f2)
                    f2.write('\n')


