#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup


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
    # text = re.sub(u"([^\u0030 -\u0039\u0041 -\u005a\u0061-\u007a])", '', text)  # 提取英文字符和数字
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


# 从头处理未处理过的数据，加入模型预处理数据时的正则
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
