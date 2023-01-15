import time
import timeit

import pandas as pd

dict_ = {}
x = dict_.get('targetColumn') if 'targetColumn' in dict_ else None
print(x is None)
x = dict_.get('targetColumn')
print(x is None)
exit()


class Record:
    def __init__(self, a, b, c) -> None:
        # self.a = 1
        # "round", "buy_money_cumsum", "buy_next", "sell"
        self.buy_time = a
        self.share = b
        self.xx = c

    @classmethod
    def obj_hook(cls, data):
        return cls(data.get("buy_time"),
                   data.get("share"), None)


class Stock:
    def __init__(self, **kwargs) -> None:
        self.code = kwargs.get("code")
        self.records = kwargs.get("records")
        self.xx = 1

    @ classmethod
    def obj_hook(cls, obj_dict):
        return cls(code=obj_dict.get("code"),
                   records=[Record.obj_hook(k) for k in obj_dict.get("records")]
                   )


x = [{"buy_time": "1", "share": 2}, {"buy_time": "1", "share": 12}]
x1 = Stock.obj_hook({"code": "sa", "records": x})
print(x1)
exit()
df = pd.DataFrame({"x": [2, 2, 3], "x1": [1, 2, 3], "x2": [1, 2, 3]})
print(df)
x = df.loc[df["x"] == 2, "x1"]
print(type(x))
df.loc[df["x"] == 2, ["x1", "x2"]] = 22
print(df)
exit()
val = max(df.loc[df["x"] == 2, "x1"])
df.loc[(df["x"] == 2) & (df.loc[df["x"] == 2, "x1"] == max(df.loc[df["x"] == 2, "x1"])), "x2"] = 2222

# df.sort_values(inplace=)
if df.loc[df["x"] == 2, "x1"].empty:
    print(1)
else:
    print(2)
# df2 = pd.DataFrame({"x": [5], "x1": [10]})
# print(df2)
exit()
# df = pd.concat([df, df2], axis=0)
# print(df)
# writer = pd.ExcelWriter('sas.xlsx')
# data_list = [{"语文": {"姓名": ["张三", "李四", "王五"], "分数": [80, 90, 60]}},
#              {"数学": {"姓名": ["张三", "李四", "王五"], "分数": [90, 80, 70]}}]
# for data in data_list:
#     for key, value in data.items():
#         my_data = pd.DataFrame(value)
#         my_data.to_excel(writer, sheet_name=key)
#         print(my_data)
# writer.close()

# df = pd.read_excel("sas.xlsx")
# print(df)

# f = pd.ExcelFile('sas.xlsx')
# for i in f.sheet_names:
#     print(i)
# #     d = pd.read_excel('./data.xlsx', sheetname=i)
# io = pd.io.excel.ExcelFile("sas.xlsx")
# for i in range(2):
#     # data =pd.read_excel(io, sheetname='持仓明细')
#     data = pd.read_excel(io, sheet_name=0, index_col=[0])
#     data2 = pd.read_excel(io, sheet_name=1, index_col=[0])
# io.close()

# print(data)
# print(data2)

# df = pd.read_csv("")
df = pd.DataFrame(index=[1, 1, 1, 2, 2, 2])
print(df)
