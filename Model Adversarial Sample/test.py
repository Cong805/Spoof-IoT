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

def preprocess(input):
    # input -> token list
    text = utils.extracTextFromHtml(input)
    text = text.replace("/", " ").replace("=", " ").replace(":", "").replace("\"", "").replace("'", " ")
    token_list = text.split()
    return token_list


if __name__ == '__main__':

    sys.stdout = Logger("adv_article_2000.log", sys.stdout)
    filelist = ["test_adv_filter.json"]

    for i, file in enumerate(filelist):
        data_path = path.raw_data_dir + '/' + file
        processed_data_path = path.processed_data_dir_b + '/' + "predict.json"

        LGS_Attack = LGS.LGS_Attack(model, pop_size=20, max_iters=20)
        with open(data_path, 'r', encoding='utf-8') as f:
            with open(processed_data_path, 'w', encoding='utf-8') as f2:
                for line in f.readlines():
                    dict = json.loads(line)
                    #if int(dict["id"])<=2000:
                         #continue
                    if int(dict["id"])>10000:
                         break
                    #if int(dict["id"]) >= 1000:
                    #    break
                    id = dict["id"]

                    print("id", id)
                    input = dict['banner'].lower()
                    #if dict['brand'] == 'AXIS' or 'FTP' in dict['banner']:
                        #print("failed")
                        #continue
                    token_list = preprocess(input)
                    banner_rm_html = ' '.join(token_list)
                    #print("token_list", token_list)

                    result = LGS_Attack.prediction(token_list)
                    device_type =result[0]
                    brand = result[1]
                    product = result[2]
                    standard_dict = {
                        "id": dict["id"],
                        'device_type': device_type,
                        'brand': brand,
                        'product': product
                    }
                    json.dump(standard_dict, f2)
                    f2.write('\n')
                    print(result)

