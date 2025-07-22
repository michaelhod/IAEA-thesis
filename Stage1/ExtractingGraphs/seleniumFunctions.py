from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from urllib.parse import quote
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path

def open_selenium(filepath: str, driver):
    local_file = Path(filepath).resolve()
    url = local_file.as_uri()
    driver.get(url)

    # explicitly wait for the page to load
    driver.implicitly_wait(10)  # seconds
    wait = WebDriverWait(driver, 5)   # up to 5 s
    wait.until(lambda d: d.execute_script("return document.readyState") == "complete")
    

def get_selenium_html(driver):
    return driver.page_source

def get_bbox(XPaths: list[str], driver):
    """
    Gets all bboxes of elements in the XPaths list from the provided HTML content.
    
    Args:
        html (str): The HTML content as a string.
        XPaths (list[str]): A list of XPath expressions to locate elements.

    Returns:
        dict: A dictionary of {xpath expression:  {x:, y:, width:, height:}}.
    """

    bboxs = {}
    for xpath in XPaths:
        elem = driver.find_element(By.XPATH, xpath)
        
        box  = driver.execute_script(
            "const r = arguments[0].getBoundingClientRect();"
            "return {x: r.x, y: r.y, width: r.width, height: r.height};", elem)
        bboxs[xpath] = box

    return bboxs