import utils
from matplotlib import pyplot

train = utils.read_data('train')
test = utils.read_data('test')


def plot_chars():
    chars = utils.char_counts(test)
    # counts = sorted(chars.values(), reverse=True)
    pyplot.scatter(range(len(chars.keys())), chars.values())
    pyplot.show()


def chars_in_one_dataset_only():
    train_chars = utils.unique_chars(train)
    test_chars = utils.unique_chars(test)
    only_in_test = test_chars - train_chars
    print(len(only_in_test))
    print(train_chars)


chars_in_one_dataset_only()
