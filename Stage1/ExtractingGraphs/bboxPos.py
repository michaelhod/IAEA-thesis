from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from urllib.parse import quote
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_bbox(html: str, XPaths: list[str]):
    """
    Gets all bboxes of elements in the XPaths list from the provided HTML content.
    
    Args:
        html (str): The HTML content as a string.
        XPaths (list[str]): A list of XPath expressions to locate elements.

    Returns:
        dict: A dictionary of {xpath expression:  {x:, y:, width:, height:}}.
    """
    # Spin up headless Chrome
    opts = Options()
    opts.add_argument("--headless=new")
    driver = webdriver.Chrome(options=opts)

    driver.get("data:text/html;charset=utf-8," + quote(html))

    # explicitly wait for the page to load
    driver.implicitly_wait(10)  # seconds
    wait = WebDriverWait(driver, 5)   # up to 5 s
    wait.until(lambda d: d.execute_script("return document.readyState") == "complete")

    bboxs = {}
    for xpath in XPaths:
        elem = driver.find_element(By.XPATH, xpath)
        
        box  = driver.execute_script(
            "const r = arguments[0].getBoundingClientRect();"
            "return {x: r.x, y: r.y, width: r.width, height: r.height};", elem)
        bboxs[xpath] = box

    driver.quit()
    return bboxs
