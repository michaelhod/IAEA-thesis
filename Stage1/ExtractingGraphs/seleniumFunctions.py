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

    js = """
    var results = {};
    for (var i = 0; i < arguments[0].length; i++) {
        var xpath = arguments[0][i];
         try {
            var elem = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
            if (elem) {
                var r = elem.getBoundingClientRect();
                results[xpath] = {x: r.x, y: r.y, width: r.width, height: r.height};
             }
         } catch(e) {}
    }
    return results;
    """
    return driver.execute_script(js, XPaths)