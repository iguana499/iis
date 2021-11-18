import pandas

data = pandas.read_csv("titanic.csv")

CLASS_NUMBER = 1
PASSENGER_CLASS = "Pclass"
SEX = "Sex"


def count_first_class_man():
    count = 0
    for i in range(len(data[PASSENGER_CLASS])):
        if (data[PASSENGER_CLASS][i]) == CLASS_NUMBER and (data[SEX][i]) == "male":
            count = count + 1
    return count


print(count_first_class_man())
