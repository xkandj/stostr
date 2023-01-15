from os import path

import pandas as pd

file_path = path.join(path.dirname(__file__), f"513050_202201_202301.csv")
file_path = path.join(path.dirname(__file__), f"2023.1.9.csv")
df = pd.read_csv(file_path, index_col=[0])
# print(df.loc[df.index[0], "high"])
df_ = df.loc[df["code"] == "588000.XSHG"]
print(df_)
df_ = df_[0]["high"]
print(df_)
# high = df_.loc[df.index[0], "high"]
# print(high)
exit()
a = 0
try:
    a = 1/0
except Exception as e:
    print(1)
a += 1
print(a)
# print(df["high"])
# import datetime

# day = pd.to_datetime(df.index[0]).day
# for date in df.index:
#     print(date)
#     day2 = pd.to_datetime(date).day
#     print(day2)
