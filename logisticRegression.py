import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit

"""
Current best
word_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  strip_accents='unicode',
                                  analyzer='word',
                                  token_pattern=r'\w{1,}',
                                  ngram_range=(1, 1),
                                  max_features=20000)

char_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  strip_accents='unicode',
                                  analyzer='char',
                                  token_pattern=r'\w{1,}',
                                  ngram_range=(1,6),
                                  max_features=30000)

public score of 0.9788
"""

# TODO Play with the values when creating the word embeddings
# Change the amount of features
# change the amount of n_grams
# try different proportions of word_vectors and char vectors
# try to use LDA and QDA with SDA optimizer
# Try to use this method on preprocessed data

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat',
               'insult', 'identity_hate']

train = pd.read_csv("data/train.csv").fillna(' ')
test = pd.read_csv("data/test.csv").fillna(' ')

train_text = train["comment_text"]
test_text = test["comment_text"]

all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  strip_accents='unicode',
                                  analyzer='word',
                                  token_pattern=r'\w{1,}',
                                  ngram_range=(1, 1),
                                  max_features=20000)

# TODO try to create the word embedding from a different text corpus, preferably a large one
word_vectorizer.fit(all_text)

train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

# TODO try to change the n_grams from 6 to 5
char_vectorizer = TfidfVectorizer(sublinear_tf=True,
                                  strip_accents='unicode',
                                  analyzer='char',
                                  token_pattern=r'\w{1,}',
                                  ngram_range=(1,6),
                                  max_features=30000)

char_vectorizer.fit(all_text)

train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)

train_features = hstack([train_word_features, train_char_features])
test_features = hstack([test_word_features, test_char_features])

losses = []
predictions = {'id': test['id']}

for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(solver="sag")

    cv_loss = np.mean(cross_val_score(classifier,
                                      train_features,
                                      train_target,
                                      cv=3,
                                      scoring="roc_auc"))

    losses.append(cv_loss)
    print('CV score for class {} is {}'.format(class_name, cv_loss))

    classifier.fit(train_features, train_target)
    predictions[class_name] = classifier.predict_proba(test_features)[:,1]

print('Total CV score is {}'.format(np.mean(losses)))
submission = pd.DataFrame.from_dict(predictions)
submission.to_csv("data/logisticSubmission.csv", index=False)


