import pandas

data = pandas.read_csv("titanic.csv")

count = 0
for i in range(len(data["Pclass"])):
    if (data["Pclass"][i]) == 1 and  (data["Sex"][i]) == "male":
        count = count + 1

print(count)
