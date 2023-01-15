
# import smtplib
# import time
# from datetime import datetime
# from email.mime.text import MIMEText
# from email.utils import formataddr

# user = "lliu606@hotmail.com"
# passwd = "lxk123456"

# from_addr = "lliu606@hotmail.com"
# to_addr = "lliu606@hotmail.com"
# smtp_srv = "smtp.live.com"

# subject = "Home Server Uptime"
# message = "The server has been online for"

# smtp = smtplib.SMTP(smtp_srv, 587)
# smtp.ehlo()
# smtp.starttls()
# smtp.ehlo()
# smtp.login(user, passwd)
# smtp.sendmail(from_addr, to_addr, message)
# smtp.quit()
# print(123)
import email
import smtplib

msg = email.message_from_string('warning')
msg['From'] = "lliu606@hotmail.fr"
msg['To'] = "lliu606@hotmail.fr"
msg['Subject'] = "helOoooOo"

s = smtplib.SMTP("smtp.live.com", 587)
s.ehlo()  # Hostname to send for this command defaults to the fully qualified domain name of the local host.
s.starttls()  # Puts connection to SMTP server in TLS mode
s.ehlo()
s.login('lliu606@hotmail.fr', 'lxk123456')

s.sendmail("lliu606@hotmail.fr", "lliu606@hotmail.fr", msg.as_string())

s.quit()
