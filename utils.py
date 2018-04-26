import pandas as pd
from collections import defaultdict
from os import path

HEADERS = [
    'id',
    'comment_text',
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]


def read_data(filename='train'):
    return pd.read_csv(path.join('data', filename) + '.csv')


def unique_chars(data):
    comments = data[HEADERS[1]]
    chars = set()
    for comment in comments:
        for char in comment:
            chars.add(char)
    print(len(chars))
    # print(chars)
    return chars


def char_counts(data):
    comments = data[HEADERS[1]]
    chars = defaultdict(lambda: 0)
    for comment in comments:
        for char in comment:
            chars[char] += 1
    return chars
