import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time

# 发送邮件通知函数
def send_email(subject, body, to_email):
    from_email = "202112081362@std.uestc.edu.cn"
    from_password = "itI2ViQzC48tcuqp"

    # 创建邮件对象
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # 邮件正文
    msg.attach(MIMEText(body, 'plain'))

    # 连接到邮件服务器
    server = smtplib.SMTP('mx-edu.icoremail.net')
    server.starttls()
    server.login(from_email, from_password)

    # 发送邮件
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

# 检查空闲公寓函数
def check_apartments(url):
    proxies = {
        "http": "127.0.0.1:10809",
        "https": "127.0.0.1:10809",
    }
    response = requests.get(url, proxies=proxies, verify=False)
    soup = BeautifulSoup(response.content, 'html.parser')
    option = soup.find('option', {'class': 'sf-level-0', 'value': 'vapaa_ja_vapautumassa'})
    if option:
        count = int(option.text.split('(')[-1].split(')')[0])
        if count > 0:
            send_email(
                "Apartment Availability Notification",
                f"There are {count} vacant apartments available.",
                "461367081@qq.com"
            )
            print(f"There are {count} vacant apartments available.")
        else:
            print("No vacant apartments.")
    else:
        print("Option not found on the page.")

# 监控页面
url = "https://www.psoas.fi/en/apartments"
# url = "https://www.google.com"

while True:
    check_apartments(url)
    time.sleep(300)  # 每5分钟检查一次
