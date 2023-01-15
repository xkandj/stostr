# import numpy as np

# # x1 = [1, 2, 3]

# # x1s = [1, 4, 9]
# # x1sm = np.array(x1s).mean()
# # x1m = np.array(x1).mean()
# # std1 = np.sqrt(x1sm-x1m**2)

# # print(std1)
# # print(np.array(x1).std())


# x1 = [1, 2, 3, 4, 5, 0, 4, 5, 6]
# x2 = [11, 12, 13]

# m = len(x1)
# n = len(x2)

# x1_mean = np.array(x1).mean()
# x1_std = np.array(x1).std()
# x2_mean = np.array(x2).mean()
# x2_std = np.array(x2).std()

# x_std = 1.0/(m+n)*np.sqrt((m+n)*(m*x1_std**2+n*x2_std**2)+m*n*(x1_mean**2+x2_mean**2-2*x1_mean*x2_mean))
# print(x_std)

# print(np.array(x1+x2).std())
# # print(np.array(x1+x2))
import datetime
import time

# from job import run_today
from apscheduler.schedulers.blocking import BlockingScheduler


def run_today():
    now = datetime.datetime.now()
    if now.minute == 39:
        # sched.shutdown(wait=False)
        sched.pause()
        print(1)
    if now.minute == 42:
        sched.resume()  # no trigger
        print(2)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def job():
    print(1)


sched = BlockingScheduler()
# 时间： 周一到周五每天早上9点25, 执行run_today
# sched.add_job(run_today, "cron",  hour="*", minute="*")
sched.add_job(job, "cron", day_of_week='mon-sun',  hour="9-12,13-23", minute="*")

#  hour='0-3'
sched.start()
