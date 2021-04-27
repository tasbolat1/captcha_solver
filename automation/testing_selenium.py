from selenium import webdriver
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException

Username = 'hasniyazov2'
Password = 'Aa12112019#'

# run this

# export PATH=$PATH:/home/crslab/Downloads
# check this on Windows


# do not set userprofile, it is working ok without it
#profile = webdriver.FirefoxProfile()
#user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
#user_agent = '/home/crslab/some_python_examples/captcha_solver/automation'

#profile.set_preference("general.useragent.override", user_agent)

opts = webdriver.FirefoxOptions()
opts.headless = True # can remove, but shit happens

driver = webdriver.Firefox(options = opts)
driver.get("https:/alan-trade.kz/#/")

time.sleep(5) # wait till website opens fully

# driver.add_cookie({"name": "key", "value": "value"})


# LOGIN starts here

username = driver.find_element_by_id("email")
password = driver.find_element_by_id("password")



username.clear()
username.send_keys(Username)
username.send_keys(Keys.RETURN)


password.clear()
password.send_keys(Password)
username.send_keys(Keys.RETURN)

driver.find_element_by_class_name("button").click()
time.sleep(2)
print('Loginge kirdi!')

# LOGIN finishes here

# Selecting starts here

driver.find_element_by_xpath("//div[@routerlink='/instruments']").click()

#time.sleep(4) # need to improve in case of delay
delay = 5
waiter = WebDriverWait(driver, delay)
start = time.time()
try:
	myElem = waiter.until(EC.text_to_be_present_in_element((By.TAG_NAME, 'app-instruments'), 'ID'))
except TimeoutException:
	print('Late')

print(time.time() - start)

heading1 = driver.find_element_by_tag_name('app-instruments')
print(heading1)

soup = BeautifulSoup(driver.page_source, features="html.parser") 

with open("scraped.txt", "w", encoding="utf-8") as f:
    f.write(str(soup))
    f.close()


# this shit works!
driver.save_screenshot("screenshot.png")


driver.close()