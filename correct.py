from multiprocessing import Pool

import numpy as np
import pandas as pd
import re
from collections import Counter

import utils


def tokens(text):
    "List all the word tokens (consecutive letters) in a text. Normalize to lowercase."
    return re.findall('[a-z]+', text.lower())


ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
PROCESS_COUNT = 10

TEXT = open('data/big.txt').read()
WORDS = tokens(TEXT)
COUNTS = Counter(WORDS)
CHARACTERS_REGEX = re.compile('\w+')


def splits(word):
    "Return a list of all possible (first, rest) pairs that comprise word."
    return [(word[:i], word[i:])
            for i in range(len(word) + 1)]


def known(words):
    "Return the subset of words that are actually in the dictionary."
    return {w for w in words if w in COUNTS}


def edits0(word):
    "Return all strings that are zero edits away from word (i.e., just word itself)."
    return {word}


def edits1(word):
    "Return all strings that are one edit away from this word."
    pairs = splits(word)
    deletes = [a + b[1:] for (a, b) in pairs if b]
    transposes = [a + b[1] + b[0] + b[2:] for (a, b) in pairs if len(b) > 1]
    replaces = [a + c + b[1:] for (a, b) in pairs for c in ALPHABET if b]
    inserts = [a + c + b for (a, b) in pairs for c in ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "Return all strings that are two edits away from this word."
    return {e2 for e1 in edits1(word) for e2 in edits1(e1)}


def correct(word):
    "Find the best spelling correction for this word."
    # Prefer edit distance 0, then 1, then 2; otherwise default to word itself.
    candidates = (known(edits0(word)) or
                  known(edits1(word)) or
                  known(edits2(word)) or
                  [word])
    return max(candidates, key=COUNTS.get)


def correct_word(regex_match):
    return correct(regex_match.group().lower())


def correct_comment(comment):
    text = comment['comment_text']
    comment['comment_text'] = CHARACTERS_REGEX.sub(correct_word, text)
    return comment


def correct_df(chunk):
    print('correcting chunk', chunk.shape)
    for index, comment in chunk.iterrows():
        print(index)
        chunk.loc[index, 'comment_text'] = CHARACTERS_REGEX.sub(correct_word, comment['comment_text'])
        correct_comment(comment)
    print('done', chunk.shape)
    return chunk


def main():
    for name in ['train_small', 'train', 'test']:
        print('reading', name)
        data = utils.read_data(name)

        print('correcting', name, 'size', data.shape)
        # data = correct_df(data)
        with Pool(processes=PROCESS_COUNT) as pool:
            processed = pool.map(correct_df, np.array_split(data, PROCESS_COUNT))
            data = pd.concat(processed)

        print('writing', name)
        data.to_csv('data/' + name + '_corrected.csv', index=False)


main()
