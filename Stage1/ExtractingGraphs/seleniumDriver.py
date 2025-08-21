# seleniumDriver.py
from selenium import webdriver
import tempfile, shutil, atexit

def driver_init(disableJS=True): #This is temporary to get the swde set to work. Make this more robust in the parallel preprocessing file
    """
    Called once per worker by ProcessPoolExecutor(initializer=driver_init).
    Creates a headless Chrome and stores it in the module-level DRIVER.
    """
    global DRIVER, TMP_PROFILE
    
    if 'DRIVER' in globals() and DRIVER is not None:
        return

    TMP_PROFILE = tempfile.mkdtemp(prefix="ch_")

    opts = webdriver.ChromeOptions()
    #opts.page_load_strategy = "eager"
    opts.add_argument("--headless=new")
    if disableJS:
        prefs = {"profile.managed_default_content_settings.javascript": 2}
        opts.add_experimental_option("prefs", prefs)
    opts.add_argument("--log-level=3")
    opts.add_argument(f"--user-data-dir={TMP_PROFILE}")
    opts.add_argument("--disk-cache-size=1048576")
    opts.add_argument("--disable-application-cache")
    opts.add_argument("--media-cache-size=0")
    DRIVER = webdriver.Chrome(options=opts)
    DRIVER.command_executor.set_timeout(60)
    if disableJS:
        DRIVER.set_network_conditions(offline=True, latency=5, throughput=0)

    # tidy-up callbacks (run when the **worker process** exits)
    atexit.register(DRIVER.quit)
    atexit.register(shutil.rmtree, TMP_PROFILE, ignore_errors=True)

def get_Driver():
    if DRIVER is None:
        raise RuntimeError("driver_init has not run in this process")
    return DRIVER

def restart_Driver(disableJS=True):
    quit_driver()
    driver_init(disableJS=disableJS)

def quit_driver():
    global DRIVER

    DRIVER.quit()
    DRIVER = None