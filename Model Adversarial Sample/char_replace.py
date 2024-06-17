#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import numpy as np

np.random.seed(2)

# 视觉相似字符对照表
similar_char_dict = {'-': '－', '9': '𝟫', '8': '𝟪', '7': '𝟩', '6': '６', '5': '𝟧', '4': '𝟦', '3': 'З', '2': '𝟸', '1': '𝟣',
             '0': '𝟢', "'": 'ʹ', ".": "．", 'a': 'а', 'b': '𝖻', 'c': 'ϲ', 'd': 'ԁ', 'e': 'е', 'f': '𝖿', 'g': '𝚐','h': 'һ',
             'i': 'і', 'j': 'ϳ', 'k': '𝗄', 'l': 'Ι', 'm': 'ｍ', 'n': '𝚗', 'o': 'о', 'p': 'р', 'q': 'ԛ', 'r': '𝚛',
             's': 'ѕ', 't': 'ｔ', 'u': 'υ', 'v': 'ν', 'w': 'ԝ', 'x': 'х', 'y': 'у', 'z': '𝗓', 'A': 'А', 'B': 'В', 'C': 'С',
             'D': 'Ｄ', 'E': 'Ε', 'F': 'Ϝ', 'G': 'Ԍ', 'H': 'Н', 'I': 'І', 'J': 'Ј', 'K': 'К', 'L': '𝖫', 'M': 'М','N': '𝖭', 'O': 'Ο', 'P': 'Р',
             'Q': 'Ԛ', 'R': '𝖱', 'S': 'Ѕ', 'T': 'Т', 'U': 'Ｕ', 'V': 'Ѵ', 'W': 'Ԝ', 'X': 'Х', 'Y': 'Ү', 'Z': 'Ζ'
             }


def invisible_char(word):
    p = np.random.randint(1, len(word))
    return word[:p] + u"\u2060" + word[p:]


def reverse_char(word):
    p1 = np.random.randint(0, len(word))
    p2 = np.random.randint(p1, len(word))
    return word[:p1] + u"\u202e" + u"\u202d" + word[p1:p2] + u"\u202c" + u"\u202c" + word[p2:]


def similar_char(word, num, similar=similar_char_dict):
    p_list = np.random.randint(0, len(word), num)
    words = []
    for p in p_list:
        if word[p] in similar:
            visual_letters = similar[word[p]]
            n = np.random.randint(0, len(visual_letters))
            words.append(word[:p] + visual_letters[n] + word[p + 1:])
    return words


def visual(word):
    # 返回视觉相似替换候选词

    # choice_n = np.random.randint(0, 3)
    # choice_n=1
    # if choice_n==0:
    #     tword1 = invisible_char(word)
    # if choice_n==1:
    #     tword2 = reverse_char(word)
    # if choice_n==2:
    #     tword3 = similar_char(word)
    words = []
    tword1 = invisible_char(word)
    #tword2 = reverse_char(word)
    #twords3 = similar_char(word, num=3)  # 选num个字符相似替换词
    words.append(tword1)
   #words.append(tword2)
    # print(twords3)
    #if not twords3:
    #    return words
    #words.extend(twords3)
    return words

# print(visual("word"))
