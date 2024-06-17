import io
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import path_config as path
import utils

if __name__ == '__main__':

    #再次处理数据

    with io.open(path.all_path,'r', encoding='utf-8') as f:
        with io.open(path.all_pro_path, 'w', encoding='utf-8') as f2:
            for line in tqdm(f):
                content, label = line.split('\t')
                text = utils.preprocessing(content)
                if len(text):
                    f2.write(' '.join(text)+'\t'+label)