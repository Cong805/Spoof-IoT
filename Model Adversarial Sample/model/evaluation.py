#!/usr/bin/env python
# coding: utf-8
import sys

sys.path.append(r"model")

import json
import os
from tqdm import tqdm
import re
# from IPython.core.debugger import set_trace
from pprint import pprint
# from transformers import AutoModel, BertTokenizerFast
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import glob
import time
from common.utils import Preprocessor
from ERModel import (HandshakingTaggingScheme,
                      DataMaker4BiLSTM,
                      TPLinkerBiLSTM,
                      MetricsCalculator)
# import wandb
# import yaml
import config
# from glove import Glove
import numpy as np

config = config.eval_config
hyper_parameters = config["hyper_parameters"]

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = str(config["device_num"])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_home = config["data_home"]
experiment_name = config["exp_name"]
test_data_path = os.path.join(data_home, experiment_name, config["test_data"])
batch_size = hyper_parameters["batch_size"]
rel2id_path = "dataset/rel2id.json"
# os.path.join(data_home, experiment_name, config["rel2id"])
save_res_dir = os.path.join(config["save_res_dir"], experiment_name)
global max_seq_len
max_seq_len = hyper_parameters["max_test_seq_len"]
sliding_len = hyper_parameters["sliding_len"]
force_split = hyper_parameters["force_split"]
word_embedding_path = "dataset/pretrained/glove_100_dataset14.emb" #config["pretrained_word_embedding_path"]
model_state_dir = config["model_state_dict_dir"]
save_res = config["save_res"]
score = config["score"]
use_type = "test"  # "test"
if use_type == "test":
    global save_path

# for reproductivity
torch.backends.cudnn.deterministic = True


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def filter_duplicates(rel_list):
    rel_memory_set = set()
    filtered_rel_list = []
    for rel in rel_list:
        rel_memory = "{}\u2E80{}\u2E80{}\u2E80{}\u2E80{}".format(rel["subj_tok_span"][0],
                                                                 rel["subj_tok_span"][1],
                                                                 rel["predicate"],
                                                                 rel["obj_tok_span"][0],
                                                                 rel["obj_tok_span"][1])
        if rel_memory not in rel_memory_set:
            filtered_rel_list.append(rel)
            rel_memory_set.add(rel_memory)
    return filtered_rel_list


def predict(data_maker, rel_extractor, test_data, split_test_data, handshaking_tagger):
    '''
    test_data: if split, it would be samples with subtext
    ori_test_data: the original data has not been split, used to get original text here
    '''
    indexed_test_data = data_maker.get_indexed_data(test_data, max_seq_len, data_type="test")  # fill up to max_seq_len
    test_dataloader = DataLoader(MyDataset(indexed_test_data),
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=0,  # note
                                 drop_last=False,
                                 collate_fn=lambda data_batch: data_maker.generate_batch(data_batch, data_type="test"),
                                 )

    pred_sample_list = []
    # for batch_test_data in tqdm(test_dataloader, desc="Predicting"):
    for batch_test_data in test_dataloader:
        if config["encoder"] in {"BiLSTM", }:
            sample_list, batch_input_ids, tok2char_span_list, _, _, _ = batch_test_data
            batch_input_ids = batch_input_ids.to(device)

        with torch.no_grad():
            if config["encoder"] in {"BiLSTM", }:
                batch_ent_shaking_outputs, \
                batch_head_rel_shaking_outputs, \
                batch_tail_rel_shaking_outputs = rel_extractor(batch_input_ids)
                # print("predicted!")

        batch_ent_shaking_tag, \
        batch_head_rel_shaking_tag, \
        batch_tail_rel_shaking_tag = torch.argmax(batch_ent_shaking_outputs, dim=-1), \
                                     torch.argmax(batch_head_rel_shaking_outputs, dim=-1), \
                                     torch.argmax(batch_tail_rel_shaking_outputs, dim=-1)


        for ind in range(len(sample_list)):
            gold_sample = sample_list[ind]
            tok_offset, char_offset = 0, 0
            if split_test_data:
                tok_offset, char_offset = gold_sample["tok_offset"], gold_sample["char_offset"]
        # return batch_ent_shaking_outputs, \
        #         batch_head_rel_shaking_outputs, \
        #         batch_tail_rel_shaking_outputs
        rel_lists=[]
        for ind in range(len(sample_list)):
            gold_sample = sample_list[ind]
            text = gold_sample["text"]
            text_id = gold_sample["id"]
            tok2char_span = tok2char_span_list[ind]
            ent_shaking_tag, \
            head_rel_shaking_tag, \
            tail_rel_shaking_tag = batch_ent_shaking_tag[ind], \
                                   batch_head_rel_shaking_tag[ind], \
                                   batch_tail_rel_shaking_tag[ind]

            tok_offset, char_offset = 0, 0
            if split_test_data:
                tok_offset, char_offset = gold_sample["tok_offset"], gold_sample["char_offset"]
            rel_list = handshaking_tagger.decode_rel_fr_shaking_tag(text,
                                                                    ent_shaking_tag,
                                                                    head_rel_shaking_tag,
                                                                    tail_rel_shaking_tag,
                                                                    tok2char_span,
                                                                    tok_offset=tok_offset, char_offset=char_offset)
            pred_sample_list.append({
                "text": text,
                "id": text_id,
                "relation_list": rel_list,
            })
            rel_lists.extend(rel_list)
        return batch_ent_shaking_outputs, \
                batch_head_rel_shaking_outputs, \
                batch_tail_rel_shaking_outputs,rel_lists
    #
    # # merge
    # text_id2rel_list = {}
    # for sample in pred_sample_list:
    #     text_id = sample["id"]
    #     if text_id not in text_id2rel_list:
    #         text_id2rel_list[text_id] = sample["relation_list"]
    #     else:
    #         text_id2rel_list[text_id].extend(sample["relation_list"])
    #
    # text_id2text = {sample["id"]: sample["text"] for sample in ori_test_data}
    # merged_pred_sample_list = []
    # for text_id, rel_list in text_id2rel_list.items():
    #     merged_pred_sample_list.append({
    #         "id": text_id,
    #         "text": text_id2text[text_id],
    #         "relation_list": filter_duplicates(rel_list),
    #     })
    #
    # return merged_pred_sample_list


def model(banner_data):
    # Load Data

    # Split

    if config["encoder"] in {"BiLSTM", }:
        tokenize = lambda text: text.split(" ")

        def get_tok2char_span_map(text):
            tokens = text.split(" ")
            tok2char_span = []
            char_num = 0
            for tok in tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1  # +1: whitespace
            return tok2char_span

    preprocessor = Preprocessor(tokenize_func=tokenize,
                                get_tok2char_span_map_func=get_tok2char_span_map)

    all_data = list([{"id":"test_0","text":banner_data}])
    data=all_data
    # test_data_dict = {}
    #
    # test_data_dict["0"] = json.loads(banner_data)
    # all_data = []
    # for data in list(test_data_dict.values()):
    #     print(test_data_dict.values())
    #     all_data.extend(data)

    max_tok_num = 0
    # for sample in tqdm(all_data, desc="Calculate the max token number"):
    for sample in all_data:
        # print(sample["text"])
        tokens = tokenize(sample["text"])
        max_tok_num = max(len(tokens), max_tok_num)

    split_test_data = False

    global max_seq_len
    if max_tok_num > max_seq_len:
        split_test_data = True
        # print("max_tok_num: {}, lagger than max_test_seq_len: {}, test data will be split!".format(max_tok_num,
        #                                                                                            max_seq_len))
    else:
        pass
        # print("max_tok_num: {}, less than or equal to max_test_seq_len: {}, no need to split!".format(max_tok_num,
        #                                                                                         max_seq_len))
    # max_seq_len = min(max_tok_num, max_seq_len)
    max_seq_len = hyper_parameters["max_test_seq_len"]

    if force_split:
        split_test_data = True
        # print("force to split the test dataset!")


    if split_test_data:
        short_data = preprocessor.split_into_short_samples(all_data,max_seq_len,
                                                                              sliding_len=sliding_len,
                                                                              encoder=config["encoder"],
                                                                              data_type="test")
        data=short_data

    # Decoder(Tagger)

    rel2id = json.load(open(rel2id_path, "r", encoding="utf-8"))
    handshaking_tagger = HandshakingTaggingScheme(rel2id=rel2id, max_seq_len=max_seq_len)

    # Dataset

    if config["encoder"] in {"BiLSTM", }:
        token2idx_path = "dataset/token2idx.json"#os.path.join(data_home, experiment_name, config["token2idx"])
        token2idx = json.load(open(token2idx_path, "r", encoding="utf-8"))
        idx2token = {idx: tok for tok, idx in token2idx.items()}

        def text2indices(text, max_seq_len):
            input_ids = []
            tokens = text.split(" ")
            for tok in tokens:
                if tok not in token2idx:
                    input_ids.append(token2idx['<UNK>'])
                else:
                    input_ids.append(token2idx[tok])
            if len(input_ids) < max_seq_len:
                input_ids.extend([token2idx['<PAD>']] * (max_seq_len - len(input_ids)))
            input_ids = torch.tensor(input_ids[:max_seq_len])
            return input_ids

        def get_tok2char_span_map(text):
            tokens = text.split(" ")
            tok2char_span = []
            char_num = 0
            for tok in tokens:
                tok2char_span.append((char_num, char_num + len(tok)))
                char_num += len(tok) + 1  # +1: whitespace
            return tok2char_span

        data_maker = DataMaker4BiLSTM(text2indices, get_tok2char_span_map, handshaking_tagger)

    # Model

    if config["encoder"] in {"BiLSTM", }:
        # random init embedding matrix
        word_embedding_init_matrix = np.random.normal(-1, 1,
                                                      size=(len(token2idx), hyper_parameters["word_embedding_dim"]))
        word_embedding_init_matrix = torch.FloatTensor(word_embedding_init_matrix)

        rel_extractor = TPLinkerBiLSTM(word_embedding_init_matrix,
                                       hyper_parameters["emb_dropout"],
                                       hyper_parameters["enc_hidden_size"],
                                       hyper_parameters["dec_hidden_size"],
                                       hyper_parameters["rnn_dropout"],
                                       len(rel2id),
                                       hyper_parameters["shaking_type"],
                                       hyper_parameters["inner_enc_type"],
                                       hyper_parameters["dist_emb_size"],
                                       hyper_parameters["ent_add_dist"],
                                       hyper_parameters["rel_add_dist"],
                                       )

    rel_extractor = rel_extractor.to(device)
    # from torchsummary import summary
    # summary(rel_extractor, (100,))
    # exit()
    # Merics

    metrics = MetricsCalculator(handshaking_tagger)

    # Prediction
    # load model state
    model_state_path="dataset/saved_dict/model_state_dict_5.pt"
    rel_extractor.load_state_dict(torch.load(model_state_path,map_location=torch.device('cpu')))
    rel_extractor.eval()
    o1,o2,o3,rel_list=predict(data_maker, rel_extractor, data, split_test_data, handshaking_tagger)
    return o1,o2,o3,rel_list
    print("o1:",o1)
    print("o2:", o2)
    print("o3:", o3)
    print("rel_list:",rel_list)
            # predict_statistics[save_path] = len([s for s in pred_sample_list if len(s["relation_list"]) > 0])




if __name__ == '__main__':
    import psutil
    import time
    import os
    model()
    start_time=time.time()
    # main()
    time_dif = time.time() - start_time
    print("Time usage:", time_dif)
    print(u'内存使用：{}M'.format(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))