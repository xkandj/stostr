
import datetime
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

import jqdatasdk as jq
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

jq.auth("15207115901", "Jj123456")


class Mail:
    def __init__(self) -> None:
        self.from_username = "xiazhou2023@163.com"
        self.from_password = "XYZYWWSBGRBGFDSM"
        self.to_username = "lliu606@hotmail.com"

    def send(self, remind_info):
        try:
            mess = ""
            title = ""
            for _, v in remind_info.items():
                mess_ = ""
                if v.get("high_value") is not None:
                    mess_ = f'卖:{v.get("high_value")}, '
                    mess = mess_ + mess
                if v.get("low_value") is not None:
                    mess_ = f'买:{v.get("low_value")}, '
                    mess = mess_ + mess

                if mess_ != "":
                    title = f'{v.get("name", "")}, {v.get("time", "")}; ' + title
                    mess = f'{v.get("name")}, ' + mess
                    mess = "\n" + mess

            msg = MIMEText(mess, "plain", "utf-8")
            msg["From"] = formataddr(["**", self.from_username])
            msg["To"] = formataddr(["**", self.to_username])
            msg["Subject"] = title

            server = smtplib.SMTP_SSL("smtp.163.com", 465)
            server.login(self.from_username, self.from_password)
            server.sendmail(self.from_username, [self.to_username,], msg.as_string())
            server.quit()
        except Exception as e:
            print(e)
            return False
        return True


class Stock:
    def __init__(self, date_time) -> None:
        self.date_time = date_time

    def get_remind_info(self, stock_dict):
        dt = self.date_time.strftime("%Y-%m-%d %H:%M:00")
        dt = "2023-01-09 14:12:00"

        df = pd.DataFrame()
        stock_list = [k for k in stock_dict.keys()]
        try:
            df = jq.get_price(stock_list, start_date=dt, end_date=dt, frequency="1m")
            print(f"time:{dt}, fetch the data from jq, data_size:{df.shape}")
        except Exception as e:
            print(f"time: {dt}, e: {e}")

        ret_dict = {}
        if df.empty == False:
            for stock in stock_list:
                df_ = df.loc[df["code"] == stock]
                if df_.empty == False:
                    high = df_.loc[df_.index[0], "high"]
                    low = df_.loc[df_.index[0], "low"]

                    threshold_dict = {}
                    if high >= stock_dict.get(stock).get("sell_threshold"):
                        threshold_dict["high_value"] = high
                    if low <= stock_dict.get(stock).get("buy_threshold"):
                        threshold_dict["low_value"] = low
                    if threshold_dict:
                        threshold_dict["name"] = stock_dict.get(stock).get("name")
                        threshold_dict["time"] = dt
                        ret_dict[stock] = threshold_dict

        return ret_dict


class StockScheduler:
    def __init__(self, stock_dict) -> None:
        self.stock_dict = stock_dict

    def job(self):
        now = datetime.datetime.now()
        remind_info = Stock(now).get_remind_info(self.stock_dict)

        if remind_info:
            ret = Mail().send(remind_info)
            if ret:
                print(f'时间：{now.strftime("%Y-%m-%d %H:%M:%S")}, 邮件发送成功')
            else:
                print(f'时间：{now.strftime("%Y-%m-%d %H:%M:%S")}, 邮件发送失败')

    def run(self):
        sched = BlockingScheduler()
        sched.add_job(self.job, "cron", day_of_week="mon-fri",  hour="9-11,13-23", minute="*")
        sched.start()


if __name__ == "__main__":
    stock_dict = {"588000.XSHG": {"name": "科创", "buy_threshold": 1.020, "sell_threshold": 1.059},
                  "513050.XSHG": {"name": "中概", "buy_threshold": 1.150, "sell_threshold": 1.169}}
    StockScheduler(stock_dict).run()
