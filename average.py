import pandas as pd


def average(csvfiles : list, output):
    """
    Do an average of the predicted values in the different csv files

    :param csvfiles: List with paths to the csv files that you are to be averaged
    :param output: Name of the output csv
    :return: returns a csv with the average of all the csv files
    """
    dataframes = []
    for csv in csvfiles:
        dataframes.append(pd.read_csv(csv))

    template = dataframes[0].copy()
    col = template.columns

    col = col.tolist()
    col.remove("id")

    length = len(csvfiles)

    for i in col:
        temp = 0
        for j in range(length):
            temp += dataframes[j][i]
        template[i] = temp / length

    template.to_csv(output, index=False)




def weighted_average(csvfiles : list, weights : list, output):
    """
    Do a weighted average of the predicted values in the different csv files

    :param csvfiles: List with paths to the csv files that you are to be averaged
    :param weights: List with integer weights corresponding to the csv files
    :param output: Name of the output csv
    :return returns a csv with the average of all the csv files
    """
    if len(csvfiles) != len(weights):
        raise ValueError("length of the csvfiles list and the weight list do not match")

    dataframes = []
    for csv in csvfiles:
        dataframes.append(pd.read_csv(csv))

    template = dataframes[0].copy()
    col = template.columns

    col = col.tolist()
    col.remove("id")

    weightsSum = sum(weights)

    for i in col:
        temp = 0
        for j in range(len(weights)):
            temp += dataframes[j][i] * weights[j]
        template[i] = temp / weightsSum

    template.to_csv(output, index=False)


if __name__ == "__main__":
    weighted_average(["data/test1.csv", "data/test2.csv"], [1, 2], "result.csv")





