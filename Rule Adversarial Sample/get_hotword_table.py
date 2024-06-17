import torch
import path_config as path
import utils
import scoring
import time
from importlib import import_module
import os


def get_hot_word(input_str, labels, model, mode='word', power=3):
    # 求重点词，power就是每个样本中选取重点词的个数

    if mode == 'word':
        words = []
        # print(input_str)
        res1 = model(input_str)
        pred1 = torch.max(res1, 1)[1].view(-1)  # label
        # print(pred1)
        # print(labels)
        losses = scoring.calculate_conf_batch(model, input_str, labels, vocab_length)  # 获得语句中的词语按重要性得分
        # print(losses)
        sorted, indices = torch.sort(losses, dim=1, descending=True)  # 按照得分从高到底排序

        inputsc = input_str[0].clone()
        # print(inputsc)

        # 取出分数高的三个单词的wordid，由其wordid取得单词
        for i in range(input_str[0].size()[0]):
            j = 0
            t = 0
            while j < power and t < input_str[0].size()[1]:
                # print(inputsc[0, indices[0][t]].item())
                if inputsc[i, indices[i][t]].item() < vocab_length:
                    wordid = inputsc[i, indices[i][t]].item()
                    words.append(list(utils.vocab.keys())[list(utils.vocab.values()).index(wordid)])
                    j += 1
                t += 1

        return words


def split_by_label(data_path, topath):
    # 将数据按类别划分成单个txt文件
    with open(data_path, 'r', encoding='UTF-8') as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            try:
                content, label = lin.split('\t')

            except ValueError:
                content = lin
                label = '-1'

            class_list = [x.strip() for x in open(path.class_path, encoding='utf-8').readlines()]
            with open(topath + '/' + str(class_list[int(label)].replace('/', '_')) + '.txt', 'a',
                      encoding='UTF-8') as outfile:
                outfile.write(lin)
                outfile.write('\n')


if __name__ == '__main__':
    # 按类别求重点词并去重

    time_start = time.time()

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    embedding = 'embedding_banner.npz'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'TextRNN_Att'

    x = import_module('models.' + model_name)
    config = x.Config(path.dataset, embedding)
    model = x.Model(config)
    # state = torch.load(config.save_path, map_location=torch.device('cpu'))
    # model = model.module
    model.load_state_dict(torch.load(config.save_path))

    model = model.to(device)

    # 按类别划分数据
    split_data_dir = path.train_split_dir
    if not os.listdir(split_data_dir):
        print("Spliting data...")
        split_by_label(path.train_path, split_data_dir)

    # 求重点词并存入文件
    print("Finding hot words...")
    vocab_length = len(utils.vocab) - 2
    filelist = os.listdir(split_data_dir)
    for file in filelist:
        data_path = split_data_dir + '/' + file
        print('Processing ', data_path)

        train_data = utils.build_dataset(config, data_path, utils.vocab)
        data_iter = utils.build_iterator(train_data, config)

        result_path = path.raw_hotwords_dir + '/' + file
        with open(result_path, 'w', encoding='utf-8') as f:
            for texts, labels in data_iter:
                words = get_hot_word(texts, labels, model=model, mode='word', power=3)
                for word in words:
                    f.write(word)
                    f.write('\n')

    print("Removing duplicate...")
    # 对每类对应的词表进行去重
    for file in filelist:
        result_path = path.raw_hotwords_dir + '/' + file
        rd_result_path = path.hotwords_dir + '/' + file
        fs_result_path = path.fs_hotwords_dir + '/' + file

        print('Processing ', result_path)
        utils.remove_duplicate(result_path, rd_result_path)
        utils.count_f(result_path, fs_result_path)

    time_end = time.time()
    print('Totally cost ', time_end - time_start)
