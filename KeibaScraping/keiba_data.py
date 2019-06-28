import os
import time
import urllib3
import certifi
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys


 # ヘッドレス・・・画面を表示せずに動作するモード
options = Options()
# 次の行をコメントアウトすると、Chromeの画面が立ち上がる
options.add_argument('--headless')
driver_path = os.getcwd() + '/chromedriver'
print(driver_path)
DRIVER = webdriver.Chrome(executable_path=driver_path, options=options)


def login():
    url = 'https://regist.sp.netkeiba.com/'
    login_id = os.environ['keiba_id']
    password = os.environ['keiba_password']

    # Chromeを起動
    # set Chrome driver path
    DRIVER.get(url)

    # login 処理
    DRIVER.find_element_by_id('id').send_keys(login_id)
    DRIVER.find_element_by_id('password').send_keys(password)
    DRIVER.find_element_by_class_name('FormItem_Submit_Btn').send_keys(Keys.ENTER)

    # soupオブジェクト
    soup = BeautifulSoup(DRIVER.page_source, 'lxml')
    print('ログインしました')

    time.sleep(1)

    # drieverのクローズ
    # DRIVER.close()
    # DRIVER.quit()


def pedigree_data():
    url = 'https://race.sp.netkeiba.com/?pid=oikiri&race_id=201905030311'
    DRIVER.get(url)
    soup = BeautifulSoup(DRIVER.page_source, 'lxml')
    time.sleep(1)

    print('Horse Name')
    for row in soup.find_all('div', class_='Horse_Name'):
        print(row.text)

    print('Training Time Data')
    for row in soup.find_all('tr', class_='TrainingTimeData'):
        for time_data in row.find_all('td'):
            print(time_data.text)


if __name__ == '__main__':
    login()
    pedigree_data()
    DRIVER.close()
    DRIVER.quit()

