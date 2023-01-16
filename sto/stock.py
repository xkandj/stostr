

import datetime
from os import path
from typing import Any, Dict, List, Optional, Union

import jqdatasdk as jq
import numpy as np
import pandas as pd


class StockStrategy:
    """股票交易策略

    principal: 本金
    total_shares: 总份额
    buy_threshold: 买入阈值
    sell_threshold: 卖出阈值
    profit: 盈利值
    profit_share: 盈利份额
    """

    def __init__(self, principal) -> None:
        self.principal = principal
        self.total_shares = 0
        self.total_consume = 0
        self.buy_space = 0.05
        self.buy_threshold = None
        self.buy_shares_coeff = 1E4
        self.sell_threshold = None
        self.profit = 0
        self.profit_share = 0
        self.once_profit_shares = 200

    def _init_buy_thre(self, price):
        """根据历史数据初始化阈值，每过一天，重新计算阈值"""
        # 第1版用第1天的数据, 阈值是根据当前价格，再高也得买
        self.buy_threshold = price

    def _compute_buy_thre(self, price=None, is_buy=False):
        #
        if self.total_shares == 0:
            self.buy_threshold = price
        elif is_buy:
            self.buy_threshold -= self.buy_space

    def _compute_sell_price(self):
        """计算卖出的单价"""
        if self.total_shares > 0:
            sell_shares = self.total_shares - self.once_profit_shares * (self.total_shares/self.buy_shares_coeff)
            return round(self.total_consume/sell_shares, 3)
        return 0

    def _compute_sell_thre(self):
        """根据历史数据获取阈值，考虑已盈利份额及佣金"""
        # 全部卖，卖的是多出的份额，一年只卖几次
        ...
        self.sell_threshold = 1.1
        # self.profit_share += self.once_profit_shares * (self.total_shares/self.buy_shares_coeff)

    def _compute_commission(self, curr_volume):
        """佣金，（买/卖）交易额的万5，最低5元"""
        min_commission = 5
        commission = curr_volume * 5E-4

        return max(min_commission, commission)

    def _buy(self, price):
        buy_shares = (self.total_shares / self.buy_shares_coeff + 1) * self.buy_shares_coeff
        curr_volume = buy_shares * price
        total_volume = self.total_consume + curr_volume + self._compute_commission(curr_volume)
        if self.buy_threshold >= price and self.principal >= total_volume:
            self.total_consume = total_volume
            self.total_shares += buy_shares
            self._compute_buy_thre(is_buy=True)

            print(f"buy...")
            print(
                f"buy_threshold, {self.buy_threshold}, price, {price}, total_consume, {self.total_consume}，total_shares, {self.total_shares}")

            return buy_shares
        return 0

    def _sell(self, buy_shares, sell_price, stock_price):
        """全部"""
        sell_shares = self.total_shares - buy_shares
        if sell_shares > 0 and sell_price <= stock_price:
            curr_volume = sell_shares * stock_price
            total_volume = curr_volume - self._compute_commission(curr_volume)
            self.profit_share += self.once_profit_shares * (sell_shares / self.buy_shares_coeff)
            self.profit += total_volume - self.total_consume
            self.total_consume = 0
            self.total_shares -= sell_shares

            print(f"sell...")
            print(
                f"profit_share, {self.profit_share}, profit, {self.profit}, total_consume, {self.total_consume}，total_shares, {self.total_shares}")

        if self.sell_threshold <= stock_price:
            # 当股票价格超过阈值
            if self.profit_share > 0:
                curr_volume = self.profit_share * stock_price
                self.profit += curr_volume - self._compute_commission(curr_volume)
                self.profit_share = 0

    def validate(self, df):
        """策略1"""
        # self._init_buy_thre(df.loc[df.index[0], "open"])
        self._compute_sell_thre()
        # 理论上每天可以进行多次买卖，但是当天买的份额要隔天卖
        # 先卖再买，故每天最多进行一次卖操作
        day_buy_shares = 0
        dt = datetime.datetime.strptime(df.index[0], "%Y-%m-%d %H:%M:%S")
        for date_str in df.index:
            dt_ = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            if (dt_ - dt).days != 0:
                dt = dt_
                day_buy_shares = 0

            print(f"date_str, {date_str}")
            sell_price = self._compute_sell_price()
            print(f"sell_price, {sell_price}")
            high_price = df.loc[date_str, "high"]
            print(f"high_price, {high_price}")
            self._sell(day_buy_shares, sell_price, high_price)

            self._compute_buy_thre(df.loc[date_str, "open"])
            low_price = df.loc[date_str, "low"]
            print(f"low_price, {low_price}")
            day_buy_shares += self._buy(low_price)
            print(f"day_buy_shares, {day_buy_shares}")


class Strategy:
    def __init__(self, df, stock_num) -> None:
        self.df = df
        self.stock_num = stock_num

    def buy(self, principal, price, records):
        share_list = []
        return share_list

    def sell(self, price, records):
        round_list = []
        return round_list

    # def _get_buy_interval(self):
    #     # 获取每次买的间隔，由历史数据确定，应该分为几档
    #     # 用昨天的数据确定买的间隔
    #     return 0.05

    # def _get_buy_baseshare(self):
    #     # 获取买入的基准份额
    #     # 由本金和历史数据确定
    #     return 1E4

    # def get_buy_info(self):
    #     threshold = 0  # curr_threshold - self._get_buy_interval
    #     share = 0  # curr_share * 2, self._get_buy_baseshare
    #     return (threshold, share)


class Record:
    def __init__(self,
                 rounds,
                 latest_round,
                 buy_time,
                 buy_price,
                 buy_share,
                 buy_money,
                 buy_money_cumsum,
                 buy_next,
                 sell,
                 sell_time,
                 sell_price,
                 sell_share,
                 finish,
                 profit_share,
                 profit) -> None:
        self.rounds = rounds
        self.latest_round = latest_round
        self.buy_time = buy_time
        self.buy_price = buy_price
        self.buy_share = buy_share
        self.buy_money = buy_money
        self.buy_money_cumsum = buy_money_cumsum
        self.buy_next = buy_next
        self.sell = sell
        self.sell_time = sell_time
        self.sell_price = sell_price
        self.sell_share = sell_share
        self.finish = finish
        self.profit_share = profit_share
        self.profit = profit

    @classmethod
    def obj_hook(cls, obj_dict):
        return cls(obj_dict.get("rounds"),
                   obj_dict.get("latest_round"),
                   obj_dict.get("buy_time"),
                   obj_dict.get("buy_price"),
                   obj_dict.get("buy_share"),
                   obj_dict.get("buy_money"),
                   obj_dict.get("buy_money_cumsum"),
                   obj_dict.get("buy_next"),
                   obj_dict.get("sell"),
                   obj_dict.get("sell_time"),
                   obj_dict.get("sell_price"),
                   obj_dict.get("sell_share"),
                   obj_dict.get("finish"),
                   obj_dict.get("profit_share"),
                   obj_dict.get("profit"))


class Stock:
    """股票类

    code: 代码
    profit: 盈利额，=每次交易差额+(盈利份额*卖出价-手续费)
    profit_share: 盈利份额
    sell_share_price: 卖出份额价格
    records: 买/卖的记录，当某次成功卖出后，则删除此记录；sell的长度由当前交易价格确定，价格越低长度越长
    position: 持仓份额
    available: 可用份额，后一天可用份额=持仓份额
    """

    def __init__(self,
                 code,
                 records) -> None:
        self.code = code
        self.records = records
        self.position = None
        self.available = None
        self.strategy = None

    @ classmethod
    def obj_hook(cls, obj_dict):
        return cls(obj_dict.get("code"),
                   [Record.obj_hook(r) for r in obj_dict.get("records")])

    def init_data(self, df, stock_num):
        position = 0
        max_record_round = 0
        for record in self.records:
            if record.finish == 1:
                position += record.profit_share
            else:
                position += record.buy_share
            max_record_round = max(max_record_round, record.rounds)
        self.max_record_round = max_record_round
        self.position = position
        self.available = self.position
        self.strategy = Strategy(df, stock_num)

    def update_position(self):
        ...

    def update_available(self):
        self.available = self.position

    def buy(self, principal, dt, price):
        share_list = self.strategy.buy(principal, price, self.records)

        # 没有记录也买
        for record in self.records:
            if record.finish == 0:
                consume += eval(record.buy_money)

        buy_shares = (self.total_shares / self.buy_shares_coeff + 1) * self.buy_shares_coeff
        curr_volume = buy_shares * price
        total_volume = self.total_consume + curr_volume + self._compute_commission(curr_volume)
        if self.buy_threshold >= price and self.principal >= total_volume:
            self.total_consume = total_volume
            self.total_shares += buy_shares
            self._compute_buy_thre(is_buy=True)

            print(f"buy...")
            print(
                f"buy_threshold, {self.buy_threshold}, price, {price}, total_consume, {self.total_consume}，total_shares, {self.total_shares}")
        # 注意修改self.records
        use_principal = 0
        return use_principal

        if trade_type == "buy":
            # df_current = pd.DataFrame({
            #     "round=round"),
            #     "buy_time=buy_time"),
            #     "buy_price=buy_price"),
            #     "buy_share=buy_share"),
            #     "buy_money=buy_money"),
            #     "buy_money_cumsum=buy_money_cumsum"),
            #     "buy_next=buy_next"),
            #     "sell=sell"),
            #     "sell_time": "",
            #     "sell_price": "",
            #     "sell_share": "",
            #     "finish": 0,
            #     "profit_share": 0,
            #     "profit": 0
            # })
            if df_record.empty == False:
                condition = df_record["round"] == v.get("round")
                df_record.loc[condition, ["buy_next", "sell"]] = ""
                # df_record = pd.concat([df_record, df_current], axis=0)

    def sell(self, dt, price):
        sell_list = self.strategy.sell(price, self.records)
        back_principal = 0
        for sell_ in sell_list:
            rounds = sell_.get("rounds")
            sell_share = sell_.get("sell_share")
            for record in self.records:
                if rounds == record.rounds:
                    if record.latest_round == 1:
                        back_principal += record.buy_money_cumsum

        return back_principal


class Corpus:
    def __init__(self,
                 file_type: str,
                 file_path: str) -> None:
        self.file_type = file_type
        self.file_path = file_path

    def _load_excel(self) -> Dict[str, pd.DataFrame]:
        sheet_names = pd.ExcelFile(self.file_path).sheet_names
        io = pd.io.excel.ExcelFile(self.file_path)
        df_dict = {}
        for sheet_name in sheet_names:
            df_dict[sheet_name] = pd.read_excel(io, sheet_name)
        return df_dict

    def _load_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path, index_col=[0])

    def _save_excel(self,
                    df_dict: Dict[str, pd.DataFrame]) -> None:
        writer = pd.ExcelWriter(self.file_path)
        for k, v in df_dict.items():
            v.to_excel(writer, sheet_name=k)
        writer.close()

    def _save_csv(self,
                  df: pd.DataFrame) -> None:
        df.to_csv(self.file_path)

    def fetch_data(self,
                   start_date: str,
                   end_date: str,
                   freq: str = "1m") -> pd.DataFrame:
        jq.auth("15207115901", "Jj123456")
        stock_list = ["588000.XSHG", "513050.XSHG"]

        # stock_list = ["588000.XSHG"]
        sd = "2022-01-11 09:00:00"
        ed = "2023-01-11 15:00:00"

        df = jq.get_price(stock_list, start_date=start_date, end_date=end_date, frequency=freq)
        print(df)

    def load_data(self) -> Optional[pd.DataFrame]:
        if self.file_type == "excel":
            return self._load_excel()
        elif self.file_type == "csv":
            return self._load_csv()
        else:
            return None

    def save_data(self,
                  df_data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> None:
        if self.file_type == "excel":
            self._save_excel(df_data)
        elif self.file_type == "csv":
            self._save_csv(df_data)


class Tool:
    @ staticmethod
    def filter_record_dict(record_dict: Dict[str, pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
        ret_dict = {}
        usecols = ["round", "buy_money_cumsum", "buy_next", "sell"]
        for k, v in record_dict.items():
            record_list = []
            if v.empty:
                record_list.append({})
            else:
                df = v.loc[v["finish"] == 0, usecols]
                # 去重
                df.drop_duplicates(subset="round", keep="last", inplace=True)
                # 组装
                for ind in df.index:
                    ser = df.loc[ind]
                    record_list.append(ser.to_dict())
            ret_dict[k] = record_list
        return ret_dict

    @ staticmethod
    def update_record_dict(record_dict: Dict[str, pd.DataFrame],
                           data_dict: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
        for k, v in data_dict.items():
            df_record = record_dict.get(k, pd.DataFrame)
            trade_type = v.get("type")
            if trade_type == "buy":
                # df_current = pd.DataFrame({
                #     "round=round"),
                #     "buy_time=buy_time"),
                #     "buy_price=buy_price"),
                #     "buy_share=buy_share"),
                #     "buy_money=buy_money"),
                #     "buy_money_cumsum=buy_money_cumsum"),
                #     "buy_next=buy_next"),
                #     "sell=sell"),
                #     "sell_time": "",
                #     "sell_price": "",
                #     "sell_share": "",
                #     "finish": 0,
                #     "profit_share": 0,
                #     "profit": 0
                # })
                if df_record.empty == False:
                    condition = df_record["round"] == v.get("round")
                    df_record.loc[condition, ["buy_next", "sell"]] = ""
                df_record = pd.concat([df_record, df_current], axis=0)
            elif trade_type == "sell":
                if df_record.empty:
                    raise ValueError(f"ERROR, code:{k}, data: {v}")
                condition = df_record["round"] == v.get("round")
                if df_record.loc[condition].empty:
                    raise ValueError(f"ERROR, code:{k}, data: {v}")
                # 更新df_record相关字段值
                df_record.loc[condition, ["buy_next", "sell", "sell_time", "sell_price", "sell_share"]] = np.nan
                df_record.loc[condition, "finish"] = 1
                # 更新最新一条记录的部分字段值
                latest_buy_time = max(df_record.loc[condition, "buy_time"])
                condition_ = condition & (df_record.loc[condition, "buy_time"] == latest_buy_time)
                df_record.loc[condition_, "sell_time"] = v.get("sell_time")
                df_record.loc[condition_, "sell_price"] = v.get("sell_price")
                df_record.loc[condition_, "sell_share"] = v.get("sell_share")
                df_record.loc[condition_, "profit_share"] = v.get("profit_share")
                df_record.loc[condition_, "profit"] = v.get("profit")
            # 排序，后续下载
            df_record.sort_values(["round", "buy_time"], inplace=True)


class TestMain:
    def __init__(self) -> None:
        self.record_file_path = path.join(path.dirname(__file__), "record.xlsx")
        self.record_dict = self._load_data("excel", self.record_file_path)
        self.df = self._load_data("csv", path.join(path.dirname(__file__), "history.csv"))

    @ staticmethod
    def _load_data(file_type: str,
                   file_path: str):
        return Corpus(file_type, file_path).load_data()

    def _load_stock_info(self):
        if path.isfile(self.info_filepath):
            with open(self.info_filepath, "r") as f:
                json_data = f.readlines()
            return json_data
        return None

    def _save_stock_info(self, json_data):
        with open(self.info_filepath, "w") as f:
            f.write(json_data)

    def _preprocess(self):
        self._init_stock()
        ...

    def _init_buy_thre(self, price):
        """根据历史数据初始化阈值，每过一天，重新计算阈值"""
        # 第1版用第1天的数据, 阈值是根据当前价格，再高也得买
        self.buy_threshold = price

    def _compute_buy_thre(self, price=None, is_buy=False):
        #
        if self.total_shares == 0:
            self.buy_threshold = price
        elif is_buy:
            self.buy_threshold -= self.buy_space

    def _compute_sell_price(self):
        """计算卖出的单价"""
        if self.total_shares > 0:
            sell_shares = self.total_shares - self.once_profit_shares * (self.total_shares/self.buy_shares_coeff)
            return round(self.total_consume/sell_shares, 3)
        return 0

    def _compute_sell_thre(self):
        """根据历史数据获取阈值，考虑已盈利份额及佣金"""
        # 全部卖，卖的是多出的份额，一年只卖几次
        ...
        self.sell_threshold = 1.1
        # self.profit_share += self.once_profit_shares * (self.total_shares/self.buy_shares_coeff)

    def _compute_commission(self, curr_volume):
        """佣金，（买/卖）交易额的万5，最低5元"""
        min_commission = 5
        commission = curr_volume * 5E-4

        return max(min_commission, commission)

    def _buy(self, price):
        buy_shares = (self.total_shares / self.buy_shares_coeff + 1) * self.buy_shares_coeff
        curr_volume = buy_shares * price
        total_volume = self.total_consume + curr_volume + self._compute_commission(curr_volume)
        if self.buy_threshold >= price and self.principal >= total_volume:
            self.total_consume = total_volume
            self.total_shares += buy_shares
            self._compute_buy_thre(is_buy=True)

            print(f"buy...")
            print(
                f"buy_threshold, {self.buy_threshold}, price, {price}, total_consume, {self.total_consume}，total_shares, {self.total_shares}")

            return buy_shares
        return 0

    def _sell(self, buy_shares, sell_price, stock_price):
        """全部"""
        sell_shares = self.total_shares - buy_shares
        if sell_shares > 0 and sell_price <= stock_price:
            curr_volume = sell_shares * stock_price
            total_volume = curr_volume - self._compute_commission(curr_volume)
            self.profit_share += self.once_profit_shares * (sell_shares / self.buy_shares_coeff)
            self.profit += total_volume - self.total_consume
            self.total_consume = 0
            self.total_shares -= sell_shares

            print(f"sell...")
            print(
                f"profit_share, {self.profit_share}, profit, {self.profit}, total_consume, {self.total_consume}，total_shares, {self.total_shares}")

        if self.sell_threshold <= stock_price:
            # 当股票价格超过阈值
            if self.profit_share > 0:
                curr_volume = self.profit_share * stock_price
                self.profit += curr_volume - self._compute_commission(curr_volume)
                self.profit_share = 0

    def validate(self, df):
        """策略1"""
        # self._init_buy_thre(df.loc[df.index[0], "open"])
        self._compute_sell_thre()
        # 理论上每天可以进行多次买卖，但是当天买的份额要隔天卖
        # 先卖再买，故每天最多进行一次卖操作
        day_buy_shares = 0
        dt = datetime.datetime.strptime(df.index[0], "%Y-%m-%d %H:%M:%S")
        for date_str in df.index:
            dt_ = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            if (dt_ - dt).days != 0:
                dt = dt_
                day_buy_shares = 0

            print(f"date_str, {date_str}")
            sell_price = self._compute_sell_price()
            print(f"sell_price, {sell_price}")
            high_price = df.loc[date_str, "high"]
            print(f"high_price, {high_price}")
            self._sell(day_buy_shares, sell_price, high_price)

            self._compute_buy_thre(df.loc[date_str, "open"])
            low_price = df.loc[date_str, "low"]
            print(f"low_price, {low_price}")
            day_buy_shares += self._buy(low_price)
            print(f"day_buy_shares, {day_buy_shares}")

    def test(self,
             principal: float,
             start_date: str) -> None:
        # record_dict = Tool.filter_record_dict(self.record_dict)
        df_train = self.df.loc[self.df.index < start_date, :]
        df_test = self.df.loc[self.df.index >= start_date, :]
        if df_test.empty:
            raise ValueError(f"测试数据时间段：{self.df.index[0]}--{self.df.index[-1]}。给定时间{start_date}无法验证策略")

        # 数据处理
        stock_dict = {}
        for k, v in self.record_dict.items():
            code = k.split(",")[1]
            stock = Stock.obj_hook({"code": code, "records": v.to_dict("records")})
            df = pd.DataFrame()
            if df_train.empty == False:
                df = df_train.loc[df_train["code"] == code, :]
            stock.init_data(df, len(stock_dict))
            stock_dict[code] = {"name": k, "stock": stock}

        # 数据验证
        for date_str in df_test.index:
            for k, v in stock_dict.items():
                condition = (df_test.index == date_str) & (df_test["code"] == k)
                df = df_test.loc[condition, ["code", "high", "low"]]
                if df.empty:
                    print(f"无此{k}的相关记录")
                    continue
                stock = v.get("stock")

                price = df["low"].values[0]
                principal -= stock.buy(principal, date_str, price)
                price = df["high"].values[0]
                principal += stock.sell(date_str, price)

                # stock.update_position()  # 此处都需要减去

                hour = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S").hour
                if hour == 15:
                    stock.update_available()
        # self._update_record()

        # 更新record
        # Corpus("excel", self.record_file_path).save_data(record_dict)


if __name__ == "__main__":
    TestMain().test(principal=10000*10, start_date="2022-01-02")
