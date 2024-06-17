import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from sklearn import preprocessing
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
import path_config as path
import utils
import random
from models import TextCNN,TextRCNN, TextRNN_Att
import pickle
import pprint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def strategy_A(word_list):
    # 随机从其他类重点词中找一个作为替换词
    tword = random.sample(word_list, 1)
    # print(tword)
    return next(iter(tword))

def strategy_B(embeddings,vocab, word, word_list):
    #从其他类且拼写相似重点词中找到一个与目标词余弦相似度最小的
    #需要other，要么复制要么都放在replace里，要么从replace里引用
    # similar_word_list = difflib.get_close_matches(word, word_list, 5, cutoff=0.6)
    if word in vocab:
        wordid = utils.vocab[word]
    else:
        tword = find_least_leven(word, word_list,3)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)
    similar_word_list = find_least_leven(word, word_list,3)
    if len(similar_word_list):
        other_hotword_embeddings_sim = generate_embed(embeddings, similar_word_list, emb_dim=100)

        a = torch.FloatTensor(embeddings[(wordid)]).unsqueeze(dim=0)

        # b = torch.FloatTensor(np.delete(other_hotword_embeddings_sim, embeddings[(wordid)], axis=0))
        b = torch.FloatTensor(other_hotword_embeddings_sim)

        similarity = get_att_dis(target=a, behaviored=b)

        # similarity = -similarity

        sort_syn, ind = torch.sort(similarity, descending=True)

        tword = similar_word_list[ind[0]]
        # tword = list(utils.vocab.keys())[list(utils.vocab.values()).index(ind[0])]
        return tword
    else:
        tword = most_largest_cos_other(embeddings,vocab, word, word_list)
        return tword

def strategy_C(word, similar, all=False):
    #视觉相似替换，将word中的某一个字母替换成视觉相似字母，返回替换后的词
    #需要替换表

    # for c in word:
    #     if c in similar:
    #         word = re.sub(c, similar[c], word)
    # return word

    if all:
        for c in word:
            if c in similar:
                visual_letters = similar[c]
                n = np.random.randint(0, len(visual_letters))
                word = re.sub(c, visual_letters[n], word)
        return word
    else:
        # print(len(word))
        s = np.random.randint(0, len(word))
        if word[s] in similar:
            visual_letters = similar[word[s]]
            n = np.random.randint(0, len(visual_letters))
            rletter = visual_letters[n]
        else:
            rletter = word[s]
        tword = word[:s] + rletter + word[s + 1:]
        return tword

def reserve_order(word):

    tword = u'\u202D' + u'\u2066'+ u'\u202E' + word[::-1] + u'\u202C' + u'\u2069' + u'\u202C'

    return tword

def repeat_insert(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)

    cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)

    insert_word = u'\u0008'.join(s for s in cos_word) + u'\u0008'
    if n == 0:
        twords = word + ' '+ u'\u0008' + insert_word
    else:
        twords = insert_word + ' '+ u'\u0008' + word
    return twords

def A_repeat_insert(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = strategy_A(word_list)
    cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)

    insert_word = u'\u0008'.join(s for s in cos_word) + u'\u0008'
    if n == 0:
        twords = tword + ' '+ u'\u0008' + insert_word
    else:
        twords = insert_word + ' '+ u'\u0008' + tword
    return twords

def B_repeat_insert(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = strategy_B(embeddings,vocab, word, word_list)
    cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)

    insert_word = u'\u0008'.join(s for s in cos_word) + u'\u0008'
    if n == 0:
        twords = tword + ' '+ u'\u0008' + insert_word
    else:
        twords = insert_word + ' '+ u'\u0008' + tword
    return twords

def C_repeat_insert(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = strategy_C(word, similar_0, all=False)
    cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)

    insert_word = u'\u0008'.join(s for s in cos_word) + u'\u0008'
    if n == 0:
        twords = tword + ' '+ u'\u0008' + insert_word
    else:
        twords = insert_word + ' '+ u'\u0008' + tword
    return twords

def A_reserve_order(word_list):
    Aword = strategy_A(word_list)
    tword = reserve_order(Aword)
    return tword

def B_reserve_order(embeddings,vocab, word, word_list):
    Bword = strategy_B(embeddings,vocab, word, word_list)
    tword = reserve_order(Bword)
    return tword

def C_reserve_order(word):
    Cword = strategy_C(word, similar_0, all=False)
    tword = reserve_order(Cword)
    return tword

def remove_char(word):
    s = np.random.randint(0,len(word))
    if len(word)>1:
        tword = word[:s] + word[s+1:]
    else:
        tword = word

    return tword

def insert_char(word):

    s = np.random.randint(0,len(word)+1)
    tword = word[:s] + chr(97+np.random.randint(0,26)) + word[s:]

    return tword

def cut_apart(word):
    # char_list = ['̮', '˗', 'ˑ', ' ۣ']
    if len(word)>1:
        s = np.random.randint(1,len(word))
        tword = word[:s] + '̮' + word[s:]
    else:
        tword = word

    return tword

def transpose(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    tword = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    return next(iter(tword))

def invisible_transpose(word):
    # tword = u'\u044f' + word #202E [::-1]
    # if len(word)>1:
    #     tword = word[:0] + u'\u202E' + word[0:]
    # else:
    #     tword = chr(97 + np.random.randint(0, 26))
    # return tword
    tword = u'\u202D' + u'\u2066' + u'\u202E' + word[::-1] + u'\u202C' + u'\u2069' + u'\u202C'
    # tword = u'\u202D' + u'\u202D' + u'\u202E' + word[::-1] + u'\u202C' + u'\u202C' + u'\u202C'
    # tword = u'\u202E' + word[::-1] + u'\u202C'
    return tword


def invisible_insert1(word): #相当于insert
    s = np.random.randint(0, len(word) + 1)
    tword = word[:s] + chr(97 + np.random.randint(0, 26)) + u'\u0008' + word[s:]
    return tword

def invisible_insert2(word):
    char_list = [u'\u200c', u'\u200d', u'\ufeff', u'\u2029']
    s = np.random.randint(0, len(word) + 1)
    tword = word[:s] + next(iter(random.sample(char_list, 1))) + word[s:]
    return tword

def visual(word, similar, all=False):
    #视觉相似替换，将word中的某一个字母替换成视觉相似字母，返回替换后的词

    # for c in word:
    #     if c in similar:
    #         word = re.sub(c, similar[c], word)
    # return word

    if all:
        for c in word:
            if c in similar:
                visual_letters = similar[c]
                n = np.random.randint(0, len(visual_letters))
                word = re.sub(c, visual_letters[n], word)
        return word
    else:
        # print(len(word))
        s = np.random.randint(0, len(word))
        if word[s] in similar:
            visual_letters = similar[word[s]]
            n = np.random.randint(0, len(visual_letters))
            rletter = visual_letters[n]
        else:
            rletter = word[s]
        tword = word[:s] + rletter + word[s + 1:]
        return tword


def visual_replace1(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = visual(word, similar_0)
    cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)
    insert_word = u'\u0008'.join(s for s in cos_word) + u'\u0008'
    if n == 0:
        twords = tword + ' '+ u'\u0008' + insert_word
    else:
        twords = insert_word + ' '+ u'\u0008' + tword
    return twords

#视觉相似字符对照表
similar_0 = {'-': '－', '9': '𝟫', '8': '𝟪', '7': '𝟩', '6': '６', '5': '𝟧', '4': '𝟦', '3': 'З', '2': '𝟸', '1': '𝟣',
               '0': '𝟢', "'": 'ʹ',".":"．", 'a': 'а', 'b': '𝖻', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝖿', 'g': '𝚐', 'h': 'һ',
               'i': 'і', 'j': 'ϳ', 'k': '𝗄', 'l': 'Ι', 'm': 'ｍ', 'n': '𝚗', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': '𝚛',
               's': 'ѕ', 't': 'ｔ', 'u': 'υ', 'v': 'ν', 'w': 'ԝ', 'x': 'х', 'y': 'у', 'z': '𝗓', 'A': 'А', 'B': 'В', 'C': 'С',
                'D': 'Ｄ', 'E': 'Ε', 'F': 'Ϝ', 'G': 'Ԍ', 'H': 'Н', 'I': 'І', 'J': 'Ј', 'K': 'К', 'L': '𝖫', 'M': 'М', 'N': '𝖭', 'O': 'Ο', 'P': 'Р',
                'Q': 'Ԛ', 'R': '𝖱', 'S': 'Ѕ', 'T': 'Т', 'U': 'Ｕ', 'V': 'Ѵ', 'W': 'Ԝ', 'X': 'Х', 'Y': 'Ү', 'Z': 'Ζ',
           }

similar_advanced = {'-': '－҃ ҄҇', '9': '９𝟫𝟿Ϥ', '8': '８𝟪𝟾', '7': '７𝟩𝟽', '6': '６𝟨𝟼бϬ', '5': '５𝟧𝟻Ҕ', '4': '４𝟦𝟺Ч', '3': 'ЗӠ𝟥𝟹３↋', '2': '２𝟤𝟸Ϩ', '1': '１𝟣𝟷ӏ',
               '0': '０𝟢𝟶О', "'": ' ҆ ＇ʹ΄',".":"．͵", 'a': 'аａ𝖺𝚊αάӑӓ', 'b': 'ｂ𝖻𝚋ЬϦ', 'c': 'сϲｃ𝖼𝚌ⅽ', 'd': 'ԁ𝖽𝚍ｄԀⅾ', 'e': 'е𝖾𝚎 ѐёҽӗｅϱ', 'f': 'ｆ𝖿𝚏ϝ', 'g': 'ｇ𝗀𝚐', 'h': 'һҺｈ𝗁𝚑ԧԦћ',
               'i': 'іｉⅰ𝗂𝚒', 'j': 'јϳｊ𝗃𝚓', 'k': '𝗄𝚔кҟқｋ', 'l': 'Ι𝗅𝚕ｌ', 'm': 'ｍⅿ𝗆𝚖м', 'n': 'ｎ𝗇𝚗ηп', 'o': 'οо𝗈𝚘ӧｏө', 'p': 'рｐ𝗉𝚙ρҏ', 'q': 'ԛ𝗊𝚚ｑ', 'r': 'ｒ𝗋𝚛',
               's': 'ѕｓ𝗌𝚜', 't': 'ｔϯτ𝗍𝚝', 'u': 'ｕ𝗎𝚞υμ', 'v': 'ν𝗏𝚟ѵｖⅴ', 'w': 'ԝ𝗐𝚠ѡｗω', 'x': 'х𝗑𝚡ӽӿｘⅹ', 'y': 'у𝗒𝚢УЎӯｙγ', 'z': 'ｚ𝗓𝚣↊', 'A': 'АＡΑ𝖠𝙰ΛӐӒ', 'B': 'ВΒ𝖡𝙱ϐβＢ', 'C': 'СϹ𝖢𝙲ⅭϾＣ',
                'D': 'ＤⅮ𝖣𝙳', 'E': 'Ε𝖤𝙴ЀＥЁӖԐ', 'F': '𝖥𝙵ϜҒＦ', 'G': 'Ԍ𝖦𝙶Ｇ', 'H': 'НＨ𝖧𝙷ӉҢ', 'I': 'ІӀӏ𝖨𝙸ⅠＩ', 'J': 'ͿЈ𝖩𝙹Ｊ', 'K': 'КＫΚ𝖪𝙺ϏҚҜҞЌ', 'L': 'Ｌ𝖫𝙻Ⅼ', 'M': 'МＭΜϺ𝖬𝙼ⅯӍ', 'N': 'Ｎ𝖭𝙽И', 'O': 'ΟО𝖮𝙾ＯӦӨ', 'P': 'РＰΡ𝖯𝙿ϷҎ',
                'Q': 'ԚＱ𝖰𝚀Ϙ', 'R': 'Ｒ𝖱𝚁Я', 'S': 'ЅＳ𝖲𝚂Ϩ', 'T': 'ТＴ𝖳𝚃ΤͲ', 'U': 'Ｕ𝖴𝚄ЦЏ', 'V': 'Ѵ𝖵𝚅ＶⅤ', 'W': 'Ԝ𝖶𝚆Ｗ', 'X': 'ХΧ𝖷𝚇ⅩχҲӼӾＸ', 'Y': 'Ү𝖸𝚈ΥϒＹ', 'Z': 'ΖＺ𝖹𝚉',
           }
# similar_0 = {'-': '˗', '9': 'Ꝯ', '8': '𐌚', '7': '𝟕', '6': 'б', '5': 'Ƽ', '4': 'Ч', '3': 'Ʒ', '2': 'ƻ', '1': 'l',
#                '0': 'O', "'": '`',".":"．", 'a': 'ɑ', 'b': 'Ь', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝚏', 'g': 'ɡ', 'h': 'հ',
#                'i': 'і', 'j': 'ϳ', 'k': 'ⲕ', 'l': 'ⅼ', 'm': 'ｍ', 'n': 'ո', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': 'ⲅ',
#                's': 'ѕ', 't': '𝚝', 'u': 'ս', 'v': 'ѵ', 'w': 'ԝ', 'x': '×', 'y': 'у', 'z': 'ᴢ', 'A': '𐊠', 'B': 'ꓐ', 'C': 'Ϲ',
#                 'D': 'ꓓ', 'E': '𑀚', 'F': '𐊇', 'G': 'Ԍ', 'H': 'Ⲏ', 'I': '𑀡', 'J': 'Ꭻ', 'K': 'Ƙ', 'L': '𑀉', 'M': 'Ϻ', 'N': 'Ⲛ', 'O': '𐌏', 'P': '𐊕',
#                 'Q': 'ℚ', 'R': 'Ʀ', 'S': 'Ş', 'T': 'Ƭ', 'U': '𐓎', 'V': 'Ѵ', 'W': 'ꓪ', 'X': '☓', 'Y': 'Ƴ', 'Z': 'Ꮓ',
#            }
# similar_1 = {'-': '—', '9': '৭', '8': 'Ȣ', '7': '𐒇', '6': 'Ⳓ', '5': 'ⴝ', '4': '𐋌', '3': 'Ɜ', '2': 'ᒿ', '1': '𐑑',
#                '0': 'ዐ', "'": 'ʾ',".":"﹒", 'a': 'Ɑ', 'b': 'ᖯ', 'c': '𐐽', 'd': 'ⅾ', 'e': 'ᥱ', 'f': 'ꬵ', 'g': 'ℊ', 'h': 'ℎ',
#                'i': 'Ꭵ', 'j': 'ᒎ', 'k': 'κ', 'l': 'ᥣ', 'm': 'ⅿ', 'n': 'ⴖ', 'o': '০', 'p': 'ⲣ', 'q': 'զ', 'r': 'ꭇ',
#                's': 'ꜱ', 't': 'ʈ', 'u': 'ᥙ', 'v': 'ⅴ', 'w': 'ꮃ', 'x': 'ⲭ', 'y': 'ꮍ', 'z': 'ⲍ', 'A': 'ꓮ', 'B': 'ℬ', 'C': 'Ⅽ',
#                 'D': 'Ꭰ', 'E': 'ⴹ', 'F': 'ꓝ', 'G': 'Ꮐ', 'H': 'ℋ', 'I': 'Ꮖ', 'J': 'Ꭻ', 'K': 'Ꮶ', 'L': 'ℒ', 'M': 'ℳ', 'N': 'Ꞑ', 'O': 'Օ', 'P': 'ꓑ',
#                 'Q': 'ⵕ', 'R': 'ℛ', 'S': 'ꕶ', 'T': 'ꔋ', 'U': '𐓎', 'V': 'Ⅴ', 'W': 'ꓪ', 'X': 'Ⅹ', 'Y': 'Ƴ', 'Z': 'Ꮓ',
#            }
# similar_2 = {'-': '╴', '9': '𐒘', '8': '𐌚', '7': '𐨬', '6': 'ⳓ', '5': 'ҕ', '4': 'Ꜭ', '3': '૩', '2': 'Ձ', '1': 'ߗ',
#                '0': '𑃰', "'": '´',".":"．", 'a': '𐓟', 'b': 'ƅ', 'c': 'ᥴ', 'd': 'ꓒ', 'e': 'ᧉ', 'f': 'ƒ', 'g': 'ց', 'h': 'Һ',
#                'i': 'ἰ', 'j': 'ڶ', 'k': 'ⲕ', 'l': '〡', 'm': 'ⴅ', 'n': 'ꞑ', 'o': 'ჿ', 'p': 'ꝑ', 'q': 'ᶐ', 'r': 'ꭈ',
#                's': 'ຣ', 't': '੮', 'u': 'ų', 'v': 'ν', 'w': 'ⱳ', 'x': 'ⅹ', 'y': 'ƴ', 'z': 'ⴭ', 'A': 'Ꭺ', 'B': 'ꕗ', 'C': 'ℂ',
#                 'D': '𑀥', 'E': 'ꓰ', 'F': '𐊇', 'G': 'ꓖ', 'H': 'Ꮋ', 'I': 'ⵊ', 'J': 'Ⱜ', 'K': 'Ⲕ', 'L': '𑀉', 'M': 'Ⅿ', 'N': 'Ⲛ', 'O': 'ⵔ', 'P': 'Р',
#                 'Q': 'ℚ', 'R': '𐊯', 'S': '𐐠', 'T': '𐊗', 'U': '𞤋', 'V': '𐰟', 'W': 'Ⱳ', 'X': 'Χ', 'Y': '𐊲', 'Z': 'Ⲍ',
#            }
# similar_advanced = {'-': '˗—ˉ౼╴╶╸╺╼╾╌═∽∼', '9': 'Ꝯ𐒘𐒛ꝯ୨⒐꧶ꔇ𐨳ᧁⳊⳋ𐐖ፃᑫ𐐋ዖ႖۹૧৭ঀ𑃓ǫ',
#              '8': '৪𐌚ꖉ𐠐Ȣȣ੪⒏𐤱Ց႘႙𐌇𐩳𐊧𐐝ꝸ୫', '7': '𝟕⒎𐒇𐨬𐓒𐓺ᒣ𐩴⌉˥⅂┑ヿ𐐑ꓶℸ𐨪',
#              '6': 'Ꮾᧈϭ𐐞⒍ꮾ𐑆ꕃ꧵ꕄⳒⳓ𐋉ᲜරճбნϬᑲБƂƃ', '5': 'ƽƼ⒌ⴝཏҔҕゟ🜪ђ', '4': 'Ꮞ⒋ꮞꜬꜭ𐋌Ꝝꝝ🜩Ꜯꜯㄐ𐨰𐰺φⳠⳡ𐨒Ч౺ሃᔦ',
#              '3': 'Ɜ౩ვᲕⳌⳍℨⳅꝪꝫᲙკ૩ᴣȝȜᦡƷЗзɜӠⳄᲳᲞӡ𐦯𐨢პ⒊𐭣𐭽꒱𞤴३Ҙʓ৬', '2': 'ᒿƨƧ⒉ƻᱮ২Ձᘖ𐭥ԶⲸⲹຂ𐨱᧓շ߶𑃦౽೭२ଌ',
#              '1': 'ߗ⒈𐰯𐑑ⅼ৷l∣〡⼁Ӏ꘡︳','0': 'oOዐOꬽ𑃰߀᱐𐌏𐒆꧰ଠᱛ𐐄⬯𐐃႐𐓂Ჿ𑀞Ⲟᦞ୦০o', "'": "``‘ʹʼʾˈ´່",".":"．﹒．·ˑ̩̣ ܂܁",
#              'a': 'ɑⱭɑ𐓟𐒷ꭤ𐓠𐒸𐐔ꬰꜹⲀⲁᥲᶏƋƌᲛ𐐀ᾳⱥãāáǎàἀἁὰάᾰᾱᾴᾶᾷἂἃἄἅἆἇ', 'b': 'ᖯߕƅ𐌜ᏏꮟⰓᲮᱠ♭𑃹Ƅᑲᶀꞗ␢ᵬƀᖲβъьЪЬ',
#              'c': 'ʗⲤꓚⲥ𐐽ꮯⅽᥴϲᏟ𐐕ℂ∁ꮳᑕςↅႠƈƇҫҪ￠ЄС́', 'd': 'ԁⅾꓒ꒯ⅆ𑀘ᏧꮷᲫɗᵭđᶁᖱᕷꁐȡɖ', 'e': 'еᥱᧉၔⅇꬲᶒⱸҼҽ᧙Ҿҿꬴⴒɇê',
#              'f': '𝚏ꬵƒ⨍ʄꞘ⨏ꝭꝬ⨎ꞙ⨑ᶂᵳᵮߓҒғſᶋ', 'g': 'ɡꞬꬶꞡǥℊ꧶𞤩ɠᶃġǵ❡ᧁ', 'h': 'հҺᏂႹꮒℎ𐱅ꚔꚕⱨꞕℏɦᏥꮵከћኪክんɧђ',
#              'i': 'іᎥꭵⅈίὶίἰἱﭐįᶖɨ႞ﭑīῑíǐῐìⲓℹⲒ', 'j': 'ϳјϳڶᒎⅉʝ⌡ꭻյȷɉ၂ຽĵǰⰬ⼅ﻝ', 'k': '𝒌κꮶⲕ𐊋ⲔꓗᛕᏦКкҚқⱪⳤƙƘҠҞꝀꝂҟꝁꝃҜꞢꞣЌќ',
#              'l': '1৷ꓲᥣⅼℓ׆∣㇑꘡〡❙⼁︳❘ⲓɭᶅȴιӀł𐰾', 'm': 'ｍⅿᶆᵯɱⴅꬺᗰᲝᨓᱬᛖꓟⲘⲙⱮⰮനᙏጠጣጢოෆຕ', 'n': 'ոꞑⴖᶇꞥᵰᱴրռղȠηⴄȵпńňǹἠἡἢἣἤἥἦἧл',
#              'o': 'о০0օ୦ᦞ᧐౦໐๐𐓪ჿꓳ𐐄ⲟ○𐌏𐓂ⱺ𐐃ꬽ០Ჿ𑀞ᱛߋơ◯ooОσōóǒò', 'p': 'рꓑꮲᏢⲢⲣᴘᴩ𐌓𐌛𐊕ᱞ𐓄ǷⰐꝒꝓᵱꝐꝑℙᵽꝥᶈᑭ𐐙⍴ᚹρየքᖰƥ', 'q': 'ԛႷզԛԛɋᶐꝗᑫવ૧۹Ɋꝙᖳᛩᕴঀ႖ʠ৭ǫƍ',
#              'r': 'ꭇꭈꞄɼ𐰮𐊄𐐊ꝻꝼꞧꝵᵲʳɍᶉɽꔵᚴɤŕřгГꝩ','s': 'ѕꜱꮪ𐨡Ჽ𐐠𐒖ꚂꚃꓢჽՏടᏚຣႽಽઽऽ𐊖ى⑀ᶊᵴۍیᎦ$ʂⱾ', 't': '𝚝ƭʈ੮ᵼꮏⱦᵵτȶţŧƫ',
#              'u': 'սսᥙꭒưµųⴎꭎⴗᶙሀчцūúǔùЦ', 'v': 'ѵᴠ∨ꮩᏙⅴⅤ𐰟ⴸ⋁νⱱꮴѴꝞꝟⰜⱴᶌ', 'w': 'ԝꮃꓪᴡᎳⱳⱲꟽ𐐎ᏔꮤพผฟฝꝠꝡᗯᙎաⴍɯƜωшщШЩ',
#              'x': '×ㄨ〤ꓫᵡⲭ╳ⅩⵝⲬⅹχ×✕Ҳ꒼ҳ✗✘⤫⤬ᶍꞳ✖ꭓꭔꭕꭖꭗꭘꭙ𐌢𐌗𐰓𑀋𑀌', 'y': 'уᎩγУꭹᎽꮍƴⴘႸⴁᲧყꭚꚏүϒЎўӲӳɥℽƔꝨꝩ', 'z': 'ᴢᴢⲍꮓⴭჍꓜᏃⲌℤⱬ𐓻𐓓ȤⱫⱿȥ႗ɀʐᱮᵶՀƶżʑᶎζ',
#              'A': 'ꓮΑ𐊠ꭺАᎪĄᎯ𐌀ꭿꬅ𐒲𐋎ΛѦȺ₳ȦΆÀÁÂÃÄÅĀĂẢẠḀӐӒᾸᾹⒶ�', 'B': 'ΒꕗᏴⲂꓐ𐊂В𐌁𐊡ℬ𐰝ꞖƁɃꞴßϐᏰ𐐚ꞵẞᛒᛔḂḄḆ₿Ⓑ', 'C': 'ϹꓚС𑀝Ꮯ𐐕ʗⲤ𐌂ⅭꞒ꒝ℂƇ𐒿𐓆ᏣҪↅÇ𞤌ር₡Ȼ₵ĆĈĊČᢷḈ₢Ⓒ',
#              'D': 'ᎠꭰꓓƊ𐓈𑀥𐓉ⅅↁⅮ𞤆𐐟𐰖𐌃ĐÐƉĎḊḒḎḌḐⒹ', 'E': 'Εꓰ𐌄𐊤ⴹᎬ𑀚ꭼ𐊆ĘȨẸḚɆẺΈĖÈЀÉÊËĒẼĔĚḘḔḖḜЁЄӖⒺ', 'F': 'ꓝ𐠰𐌅𐊇𐊥ƑꞘℱӺᚩᚪᚫϜᚨḞΓ₣Ⓕ',
#              'G': 'ԌꓖᏀᏩᏵᎶƓᏳǤḠꞠ₲ĢĠǴĜĞǦⒼꮆꮐꮹ', 'H': 'НꓧℋⲎꞪꖾℍᎻꀿꃅ𐋏ΗҢ𑀅ӉⱧҤᚺዘዚḨḢḤḪӇꜦ𒀂ዝਮમዙԊǶΉĤĦዜḦⒽ', 'I': 'Ꮖ𐌆Ιꓲ𐰾𑁇ⵏ⌶𑀡𐊈ⵊⅠꀤІӏᛁĮỈỊİḬÌÍÎÏĨĪĬΊΪḮЇῘῙⒾ',
#              'J': 'ᎫꓙЈꭻꞲਹ𑀮ĴຽⰬⒿ', 'K': 'ΚꓗKᏦᛕⲔꗣϏҚⱩҜҞꝀꝂꝄҠ₭ꞢĶḲƘКḴЌḰǨӃ𐒼Ⓚ', 'L': 'ᏞⅬ𑀉𐐛𐌋ꓡℒ꒒ꝆḶ⌊ꞭḺḼŁȽⱠⱢĿĹĻĽᢺḸⓁꮮ', 'M': 'ϺꓟМⅯ𐌑ⲘΜᎷⱮᛖ𐰡ℳ𐊰ꙦӍṂᛗḾṀⓂꮇ',
#              'N': 'ΝꓠⲚꞐ𐊪ŊƝℕͶИŅÑŃŇǸ₦ꞤⓃ', 'O': '0Օ𐌏ꓳⲞⵔ𐐄𐓂ᲿОᱛዐΟꙨʘΘᎾⲐӨѲᏫƠỌÒÓÔÕÖŌŎŐΌᎤṌṎṐṒӦⓄ', 'P': 'ΡꓑР𐌓𐊕ℙⲢꝐꝒƤקᏢ𐰙ᚹⱣṔṖǷҎ₱₽Ⓟ',
#              'Q': 'ԚⵕℚꝖꝘ𐌒ǪǬϘⓆ', 'R': 'ᏒƦ𐒴𐊯ꭱꓣꭆᎡᚱɌℛℜℝⱤ℞Ꞧ℟ṚŖṞŔŘṘṜⓇꮢ', 'S': 'Ѕꕶ𐐠ꚂꓢᏚꗟ꒚ᲽṨꕷŞⱾꞨᎦṢṠŚŜŠṤṦȘ₷Ⓢ',
#              'T': 'Τ𐌕ꔋ𐊗ꭲꓔ𐊱ⲦТͲ꓄ҬᎢͳƬƮȾṬŢŦṮṰṪŤȚ₮Ⓣ', 'U': '𐓎𞤋ꓴƯƲህŲɄỤṲṴṶỦÙÚÛÜŨṸṺỰỨỪỬȖⓊ', 'V': 'Ꮩ𐰟ⅤᏤ𐠨ѴṾṼѶꝞ℣Ⓥꮩ',
#              'W': 'ꓪᎳꞶᏔⱲꝠẈẆẀẂŴẄꮃ₩ⓌⰞ', 'X': 'ΧⲬꞳ𐰓ꓫⅩ☓✕𐊴꒼ꭓꭔꭕχӼӾХҲᚷᲯẊẌⓍ', 'Y': 'ΥꓬⲨ𐌖ϒҮ𐰠Ƴ𐊲ꝨỴẎҰɎץÝỲŶỶỸŸΎΫϓϔƔȲῨῩⓎ',
#              'Z': 'ΖᏃꓜⲌ႗Ȥ𐓓ⱫℤꙀꙂƵⱿẒẔŻŹŽẐⓏꮓ'}


default_filter = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


default_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"

def replace_insert(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = most_largest_cos_other(embeddings, vocab, word, word_list)
    insert_word = invisible_insert1(word)
    if n == 0:
        twords = tword + ' ' + insert_word
    else:
        twords = insert_word + ' ' + tword
    return twords

def replace_insert_sim(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = most_largest_cos_other_sim(embeddings, vocab, word, word_list)
    # insert_word = invisible_insert1(word)
    insert_word = tword
    # insert_word = next(iter(find_most_similiar(word, word_list, 0.2)))
    if n == 0:
        twords = tword + ' ' + insert_word
    else:
        twords = insert_word + ' ' + tword
    return twords

def invisible_insert1_replace1(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = invisible_insert2(word)
    cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)
    insert_word = u'\u0008'.join(s for s in cos_word) + u'\u0008'
    if n == 0:
        twords = tword + ' ' + u'\u0008' + insert_word
    else:
        twords = insert_word + ' ' + u'\u0008' + tword
    return twords

def invisible_transpose_replace1(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = invisible_transpose(word)
    cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)
    insert_word = u'\u0008'.join(s for s in cos_word) + u'\u0008'
    if n == 0:
        twords = tword + ' ' + u'\u0008' + insert_word
    else:
        twords = insert_word + ' ' + u'\u0008' + tword
    return twords

def encode(s):
    return ''.join([bin(ord(c)).replace('0b', '') for c in s])

def invisible_insert1_replace2(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = invisible_insert1(word)
    # cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)
    insert_word = u'\u0008' + encode(tword).replace('0', u'\u200d').replace('1', u'\u200c')
    if n == 0:
        twords = tword + ' ' + insert_word
    else:
        twords = insert_word + ' ' + tword
    return twords

def invisible_transpose_replace2(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = invisible_transpose(word)
    # cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)
    insert_word = u'\u0008' + encode(tword).replace('0', u'\u200d').replace('1', u'\u200c')
    if n == 0:
        twords = tword + ' ' + insert_word
    else:
        twords = insert_word + ' ' + tword
    return twords


def find_most_similiar(word, word_list, similarity=0.6):
    #找到拼写上相似的词，是字符串拼写上的，eg.由v100r001找到v100r003
    tword = difflib.get_close_matches(word, word_list, 1, cutoff=similarity)

    return tword

import Levenshtein
# def find_least_leven(word, word_list):
#     #找到编辑距离最小的词
#     # sub_list = difflib.get_close_matches(word, word_list, cutoff=similarity)
#     dis = [Levenshtein.ratio(word,word1) for word1 in word_list]
#     tword = word_list[dis.index(max(dis))]
#
#     return tword

def find_least_leven(word, word_list, number=1, distance=3):
    #找到编辑距离最小的词
    # sub_list = difflib.get_close_matches(word, word_list, cutoff=similarity)
    dis = {}
    for word1 in word_list:
        if Levenshtein.distance(word, word1) <= distance:
            dis[word1] = Levenshtein.distance(word, word1)
    # dis = [Levenshtein.ratio(word,word1) for word1 in word_list]
    if len(dis) > 0:
        M = sorted(dis.items(), key=lambda x: x[1])[:number]
        # print(M)
        # tword = word_list[dis.index(max(dis))]
        if number == 1:
            for m in M:
                tword = m[0]
        else:
            tword = []
            for m in M:
                tword.append(m[0])
    else:
        tword = []

    return tword

def leven_replace1(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = find_least_leven(word, word_list)
    cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)
    insert_word = u'\u0008'.join(s for s in cos_word) + u'\u0008'
    if n == 0:
        twords = tword + ' ' + u'\u0008' + insert_word
    else:
        twords = insert_word + ' ' + u'\u0008' + tword
    return twords

def cossim_replace1(embeddings,vocab, word, word_list):
    n = np.random.randint(0, 2)
    tword = most_largest_cos_other_sim(embeddings, vocab, word, word_list)
    cos_word = most_largest_cos_other_sim(embeddings, vocab, word, word_list)
    # cos_word = invisible_insert1(word)
    insert_word = u'\u0008'.join(s for s in cos_word) + u'\u0008'
    if n == 0:
        twords = tword + ' ' + u'\u0008' + insert_word
    else:
        twords = insert_word + ' ' + u'\u0008' + tword
    return twords

def random_select(word_list):
    #随机从其他类重点词中找一个作为替换词
    tword = random.sample(word_list, 1)

    return tword

#以下三个函数是求余弦相似度的
def get_att_dis(target, behaviored):

        attention_distribution = []

        for i in range(behaviored.size(0)):
            attention_score = torch.cosine_similarity(target, behaviored[i].view(1, -1))  # 计算每一个元素与给定元素的余弦相似度
            attention_distribution.append(attention_score)
        attention_distribution = torch.Tensor(attention_distribution)

        return attention_distribution / torch.sum(attention_distribution, 0)  # 标准化

def find_syns(embeddings, wordid):


        a = torch.FloatTensor(embeddings[(wordid)]).unsqueeze(dim=0)
        # print(wordid)

        b = torch.FloatTensor(np.delete(embeddings, wordid, axis=0))


        similarity = get_att_dis(target=a, behaviored=b)
        return similarity

def most_largest_cos_all(embeddings, word):
    #从所有所有词中找到一个与目标词余弦相似度最大的
    wordid = utils.vocab[word]
    sort_syn, ind = torch.sort(find_syns(embeddings, wordid), descending=True)

    # print(list(utils.vocab.keys())[list(utils.vocab.values()).index(wordid)])
    # return torch.FloatTensor(embeddings[(ind[0])]).unsqueeze(dim=0)
    # return ind[0]
    tword = list(utils.vocab.keys())[list(utils.vocab.values()).index(ind[0])]
    return tword


def generate_embed(embeddings, word_list, emb_dim=100):
    """根据其他类重点词词表成词向量矩阵"""
    other_hotword_embeddings = np.random.rand(len(word_list), emb_dim)

    for idx in range(len(word_list)):
        # print(embeddings[utils.vocab[word_list[idx]]])
        other_hotword_embeddings[idx, :] = embeddings[(utils.vocab[word_list[idx]])]  # 词向量矩阵
    # print(other_hot_word_embeddings)
    # np.savez_compressed(path.embed_path, embeddings=embeddings)
    return other_hotword_embeddings

def most_largest_cos_other(embeddings,vocab, word, word_list):
    #从其他类重点词中找到一个与目标词余弦相似度最大的
    if word in vocab:
        wordid = utils.vocab[word]
    else:
        tword = find_least_leven(word, word_list)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)
    other_hotword_embeddings = generate_embed(embeddings, word_list, emb_dim=100)

    a = torch.FloatTensor(embeddings[(wordid)]).unsqueeze(dim=0)

    # b = torch.FloatTensor(np.delete(other_hotword_embeddings, embeddings[(wordid)], axis=0))
    b = torch.FloatTensor(other_hotword_embeddings)
    # print(word)
    # print(wordid)

    similarity = get_att_dis(target=a, behaviored=b)

    similarity = -similarity

    sort_syn, ind = torch.sort(similarity)
    # print(similarity)
    # print(sort_syn)
    # print(ind)

    tword = word_list[ind[0]]
    # tword = list(utils.vocab.keys())[list(utils.vocab.values()).index(ind[0])]
    return tword

def most_largest_cos_other_sim(embeddings,vocab, word, word_list):
    #从其他类且拼写相似重点词中找到一个与目标词余弦相似度最大的
    # similar_word_list = difflib.get_close_matches(word, word_list, 5, cutoff=0.6)
    if word in vocab:
        wordid = utils.vocab[word]
    else:
        # tword = find_most_similiar(word, word_list)
        tword = find_least_leven(word, word_list)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)
    similar_word_list = find_least_leven(word, word_list,5)
    if len(similar_word_list):
        other_hotword_embeddings_sim = generate_embed(embeddings, similar_word_list, emb_dim=100)

        a = torch.FloatTensor(embeddings[(wordid)]).unsqueeze(dim=0)

        # b = torch.FloatTensor(np.delete(other_hotword_embeddings_sim, embeddings[(wordid)], axis=0))
        b = torch.FloatTensor(other_hotword_embeddings_sim)

        similarity = get_att_dis(target=a, behaviored=b)

        similarity = -similarity

        sort_syn, ind = torch.sort(similarity)

        tword = similar_word_list[ind[0]]
        # tword = list(utils.vocab.keys())[list(utils.vocab.values()).index(ind[0])]
        return tword
    else:
        tword = most_largest_cos_other(embeddings,vocab, word, word_list)
        return tword

#从其他类单词中找出梯度方向投影距离最大的（仅限基于模型白盒）,texts未经raw处理
def most_largest_projection(word_index, model, texts, embeddings, vocab, word, word_list):

    if word in vocab:
        wordid = utils.vocab[word]
    else:
        tword = find_most_similiar(word, word_list)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)
    input_seq, f_label= utils.build_text(texts,vocab)

    # word_index = input_seq.index(utils.vocab[word])


    input_seq = torch.Tensor(input_seq).long().view(1, -1).unsqueeze(dim=0)
    if device:
        input_seq = input_seq.to(device)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    embd, output = model(input_seq, returnembd=True)
    tlabel = torch.max(output, 1)[1].view(-1)

    embd.retain_grad()

    loss = F.nll_loss(output, torch.Tensor([float(tlabel)]).long().to(device))
    loss.backward()


    if len(word_list):

        syn = []
        for i in range(len(word_list)):

            y= embd.grad[0][word_index]

            x= embeddings[utils.vocab[word_list[i]]]

            # projection = y * np.dot(x, y) / np.dot(y, y)
            #
            # projection = torch.norm(projection, p=2).item()
            y_square = torch.norm(y, p=2).item()
            distance = (x - embeddings[wordid]) * y_square
            distance = torch.norm(torch.FloatTensor(distance), p=2).item()

            syn.append(distance)


        tword = word_list[syn.index(max(syn))]
        return tword
    else:
        tword = find_most_similiar(word, word_list)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)

def get_same_element_index(ob_list, word):

    return [i+1 for (i, v) in enumerate(ob_list) if v == word]
#从其他类、拼写相似的单词中找出梯度方向投影距离最大的（仅限基于模型白盒）,texts未经raw处理
def most_largest_projection_sim(word_index, model, texts, embeddings, vocab, word, word_list):

    if word in vocab:
        wordid = utils.vocab[word]
    else:
        tword = find_most_similiar(word, word_list)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)
    input_seq, f_label= utils.build_text(texts,vocab)
    # word_index = input_seq.index(utils.vocab[word])
    # word_index_list = get_same_element_index(input_seq, utils.vocab[word])


    input_seq = torch.Tensor(input_seq).long().view(1, -1).unsqueeze(dim=0)
    if device:
        input_seq = input_seq.to(device)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    # model.train()
    banner = input_seq.clone()
    # print(banner)
    embd, output = model(input_seq, returnembd=True)
    tlabel = torch.max(output, 1)[1].view(-1)
    # print(embeddings[utils.vocab[word]])
    embd.retain_grad()

    loss = F.nll_loss(output, torch.Tensor([float(tlabel)]).long().to(device))
    loss.backward()
    # print(embd.grad[0])
    # y= embd.grad[0][word_index]
    # x= embeddings[utils.vocab['c1000z']]
    # projection = y * np.dot(x, y) / np.dot(y, y)
    # projection = torch.norm(projection, p=2)
    similar_word_list = difflib.get_close_matches(word, word_list, 5, cutoff=0.6)
    if len(similar_word_list):
        # syn = np.random.rand(len(similar_word_list),1)
        syn = []

        for i in range(len(similar_word_list)):
            # print(i)
            y= embd.grad[0][word_index]

            x= embeddings[utils.vocab[similar_word_list[i]]]

            # projection = y * np.dot(x, y) / np.dot(y, y)
            #
            # projection = torch.norm(projection, p=2).item()

            # x_square = torch.norm(torch.FloatTensor(x), p=2).item()
            y_square = torch.norm(y, p=2).item()
            # distance = (x_square - projection)/x_square * y_square
            distance = (x - embeddings[wordid]) * y_square
            distance = torch.norm(torch.FloatTensor(distance), p=2).item()
            # print(distance)
            # syn[i, :] = projection
            syn.append(distance)

        # sort_syn, ind = torch.sort(torch.FloatTensor(syn))
        # print(syn)
        # print(similar_word_list[syn.index(max(syn))])
        tword = similar_word_list[syn.index(max(syn))]
        # print(word_index)
        # print(tword)
        return tword
    else:
        tword = most_largest_projection(word_index, model, texts, embeddings, vocab, word, word_list)
        return tword
#结合余弦值和投影向量的长度：余弦值和投影向量的长度都要尽量小
#从其他类词中选
def cos_proj_combine(word_index, model, texts, embeddings, vocab, word, word_list):
    if word in vocab:
        wordid = utils.vocab[word]
    else:
        tword = find_most_similiar(word, word_list)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)

    input_seq, f_label = utils.build_text(texts, vocab)

    # word_index = input_seq.index(utils.vocab[word])

    input_seq = torch.Tensor(input_seq).long().view(1, -1).unsqueeze(dim=0)
    if device:
        input_seq = input_seq.to(device)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    embd, output = model(input_seq, returnembd=True)
    tlabel = torch.max(output, 1)[1].view(-1)

    embd.retain_grad()

    loss = F.nll_loss(output, torch.Tensor([float(tlabel)]).long().to(device))
    loss.backward()


    if len(word_list):
        # syn = np.random.rand(len(similar_word_list),1)
        syn = []
        for i in range(len(word_list)):
            # print(i)
            y = embd.grad[0][word_index]

            x = embeddings[utils.vocab[word_list[i]]]
            x = x - embeddings[wordid]
            y = y.cpu()
            y_square = torch.norm(y, p=2).item()

            # projection = y * np.dot(x, y) / np.dot(y, y)
            #
            # projection = torch.norm(projection, p=2).item()
            # print(projection)
            a = np.dot(x.dot(y), y) / y_square
            projection = x - a
            # projection = torch.norm(projection, p=2).item()
            projection = np.linalg.norm(projection)

            cos = x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
            # print(type(cos))

            distance = projection * np.sign(cos)
            # print(cos)
            # distance = torch.norm(torch.FloatTensor(distance), p=2).item()

            syn.append(distance)


        tword = word_list[syn.index(min(syn))]
        return tword
    else:
        tword = find_most_similiar(word, word_list)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)


#从其他类词中拼写相似的里面选
def cos_proj_combine_sim(word_index,model, texts, embeddings,vocab, word, word_list):
    # similar_word_list = difflib.get_close_matches(word, word_list, 5, cutoff=0.6)
    # print(similar_word_list)
    if word in vocab:
        wordid = utils.vocab[word]
    else:
        tword = find_most_similiar(word, word_list)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)

    similar_word_list = difflib.get_close_matches(word, word_list, 5, cutoff=0.6)
    input_seq, f_label = utils.build_text(texts, vocab)

    # word_index = input_seq.index(utils.vocab[word])

    input_seq = torch.Tensor(input_seq).long().view(1, -1).unsqueeze(dim=0)
    if device:
        input_seq = input_seq.to(device)

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    embd, output = model(input_seq, returnembd=True)
    tlabel = torch.max(output, 1)[1].view(-1)

    embd.retain_grad()

    loss = F.nll_loss(output, torch.Tensor([float(tlabel)]).long().to(device))
    loss.backward()


    if len(similar_word_list):
        # syn = np.random.rand(len(similar_word_list),1)
        syn = []
        for i in range(len(similar_word_list)):
            # print(i)
            y = embd.grad[0][word_index]

            x = embeddings[utils.vocab[similar_word_list[i]]]
            x = x - embeddings[wordid]
            y = y.cpu()
            y_square = torch.norm(y, p=2).item()


            a = np.dot(x.dot(y), y) / y_square
            projection = x - a
            # projection = torch.norm(projection, p=2).item()
            projection = np.linalg.norm(projection)

            # projection = y * np.dot(x, y) / np.dot(y, y)
            #
            # projection = torch.norm(projection, p=2).item()
            # print(projection)

            cos = x.dot(y)/(np.linalg.norm(x)*np.linalg.norm(y))
            # print(cos)

            distance = projection * np.sign(cos)
            # distance = torch.norm(torch.FloatTensor(distance), p=2).item()

            syn.append(distance)


        tword = similar_word_list[syn.index(min(syn))]
        return tword
    else:
        tword = cos_proj_combine(word_index,model, texts, embeddings,vocab, word, word_list)
        return tword



#从其它类单词中找出对置信度影响最大的词,（黑盒可以用？存疑）,texts未经raw处理,其他类的词表长时会很慢
def most_largest_conf(word_index,model, texts,vocab, word, word_list):
    word_id_list = []
    for i in range(len(word_list)):
        word_id_list.append(utils.vocab[word_list[i]])

    input_seq, f_label = utils.build_text(texts,vocab)

    # word_index = input_seq.index(utils.vocab[word])

    input_seq = torch.Tensor(input_seq).long().view(1, -1).unsqueeze(dim=0)
    if device:
        input_seq = input_seq.to(device)
    res = model(input_seq)

    tlabel = torch.max(res, 1)[1].view(-1)

    if len(word_list):
        syn = []
        tempinputs = input_seq.clone()

        for word_id in word_id_list:

            tempinputs[0] = torch.LongTensor([word_id if i==word_index else a for a in input_seq[0][0]])

            with torch.no_grad():
                tempoutput = model(tempinputs)
            syn.append(F.cross_entropy(tempoutput, tlabel, reduction='none').item())
        # print(syn)
        tword = word_list[syn.index(max(syn))]
        return tword


#从其它类且拼写相似的单词中找出对置信度影响最大的词,（黑盒可以用？存疑）,texts未经raw处理
def most_largest_conf_sim(word_index, model, texts, vocab, word, word_list):
    similar_word_list = difflib.get_close_matches(word, word_list, 5, cutoff=0.6)
    similar_word_id_list = []
    for i in range(len(similar_word_list)):
        similar_word_id_list.append(utils.vocab[similar_word_list[i]])

    input_seq, f_label = utils.build_text(texts,vocab)

    # word_index = input_seq.index(utils.vocab[word])

    input_seq = torch.Tensor(input_seq).long().view(1, -1).unsqueeze(dim=0)
    if device:
        input_seq = input_seq.to(device)
    res = model(input_seq)

    tlabel = torch.max(res, 1)[1].view(-1)

    if len(similar_word_list):
        syn = []
        tempinputs = input_seq.clone()

        for similar_id in similar_word_id_list:

            tempinputs[0] = torch.LongTensor([similar_id if i==word_index else a for a in input_seq[0][0]])

            with torch.no_grad():
                tempoutput = model(tempinputs)
            syn.append(F.cross_entropy(tempoutput, tlabel, reduction='none').item())
        # print(syn)
        tword = similar_word_list[syn.index(max(syn))]
        return tword
    else:
        tword = find_most_similiar(word, word_list)
        if tword:
            return next(iter(tword))
        else:
            return visual(word, similar_0)






if __name__ == '__main__':

    torch.manual_seed(8)
    torch.cuda.manual_seed(8)

    embedding = 'embedding_banner.npz'
    # dataset = 'dataset-dt'
    dataset = 'dataset'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextRNN_Att.Model(TextRNN_Att.Config(dataset, embedding))
    state = torch.load(path.p_rnnatt_modelpath, map_location=torch.device('cpu'))
    model = model.to(device)

    try:
        model.load_state_dict(state['state_dict'])
    except:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(state['state_dict'])
        model = model.module
    print('Type input:')
    info = np.load(dataset + '/data/' + embedding, allow_pickle=True)
    # print(info.files)
    embeddings = info['embeddings'][()]
    vocab = utils.vocab
    # print(vocab['dvrh264'])
    # ind = most_largest_cos(embeddings, 'c1000z')
    # print(ind)
    # print(list(utils.vocab.keys())[list(utils.vocab.values()).index(ind)])
    # print(utils.vocab['c1000z'])
    # print(embeddings)
    # word_list = ['c1000z', 'vxworks5.4.2', '200', 'http','cl','1000']


    input = '5.4 thttpd dvrh264 25b last-modified bytes dvrh264 activex thttpd'
    input1 = input
    text = utils.preprocessing(input)
    texts = ' '.join(text)
    label = '66'
    hotwords = ['thttpd','5.4','dvrh264']
    def dictlist(filepath):
        # 将重点词表的txt转成列表型式
        dicts = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
        return dicts
    word_list = dictlist('hotwords/grad_p_att_hotwords.txt')
    word_list.remove('thttpd')
    # word_list.remove('5.4')
    word_list.remove('dvrh264')
    # word_list = ['h267a', 'h2c','shop', 'sharp', 'happy', 'th', 'sh']
    # print(most_largest_cos_other(embeddings, 'c1000z',word_list))
    # print(generate_embed(embeddings, word_list))
    # print(embeddings[552])
    # print(embeddings[utils.vocab['dvrh264']])
    import time

    time_start = time.time()

    hhh = []
    for word in text:
        if word in hotwords:
            hhh.append(word)
    # print(hhh)
    for word in set(hhh):
        word_index_list = get_same_element_index(text, word)
        for word_index in word_index_list:
        # tword = most_largest_projection(model, texts,label, embeddings, word, word_list)
        #     tword = most_largest_cos_other_sim(embeddings,vocab, word, word_list)
        #     input = re.sub(word, tword, input,flags=re.IGNORECASE, count=1)
            tword = most_largest_conf_sim(word_index,model, texts, word, word_list)
            input = re.sub(word, tword, input, flags=re.IGNORECASE, count=1)

            tword1 = cos_proj_combine_sim(word_index,model, texts, embeddings,vocab, word, word_list)
            input1 = re.sub(word, tword1, input1,flags=re.IGNORECASE, count=1)
    print(input)
    print(input1)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    # print(difflib.get_close_matches('hp', word_list, 5, cutoff=0.5))
