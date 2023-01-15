import datetime
from os import path

import jqdatasdk as jq
import numpy as np
# https://www.joinquant.com/help/api/help#api:API%E6%96%87%E6%A1%A3
# https://www.joinquant.com/help/api/help#JQData:JQData
import pandas as pd

jq.auth("15207115901", "Jj123456")

stock_list = ["588000.XSHG", "513050.XSHG"]
# stock_list = ["588000.XSHG"]
sd = "2022-01-11 09:00:00"
ed = "2023-01-11 15:00:00"

df = jq.get_price(stock_list, start_date=sd, end_date=sd, frequency="1m")
print(df)
if not df.empty:
    df.to_csv(path.join(path.dirname(__file__), f"stock_{sd[0:4]}{sd[5:7]}_{ed[0:4]}{ed[5:7]}.csv"), index=False)


df = pd.DataFrame()
# stock_list = [k for k in stock_dict.keys()]

exit()
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:00")
print(now)
for stock in stock_list:
    df = jq.get_price(stock, start_date=sd, end_date=sd, frequency="1m")
    print(df.loc[df.index[0], "high"])

    ...
    # df.to_csv(path.join(path.dirname(__file__),f"{stock.split('.')[0]}_{sd[0:4]}{sd[5:7]}_{ed[0:4]}{ed[5:7]}.csv"))
