# seleniumDriver.py
from selenium import webdriver
import tempfile, shutil, atexit

def driver_init(timoutOccured=True): #This is temporary to get the swde set to work. Make this more robust in the parallel preprocessing file
    """
    Called once per worker by ProcessPoolExecutor(initializer=driver_init).
    Creates a headless Chrome and stores it in the module-level DRIVER.
    """
    global DRIVER, TMP_PROFILE
    DRIVER = None
    if DRIVER is not None:          # already initialised in this process
        return

    TMP_PROFILE = tempfile.mkdtemp(prefix="ch_")

    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument(f"--user-data-dir={TMP_PROFILE}")
    opts.add_argument("--disk-cache-size=1048576")
    opts.add_argument("--disable-application-cache")
    opts.add_argument("--media-cache-size=0")
    DRIVER = webdriver.Chrome(options=opts)
    if timoutOccured:
        DRIVER.set_network_conditions(offline=True, latency=5, throughput=0)

    # tidy-up callbacks (run when the **worker process** exits)
    atexit.register(DRIVER.quit)
    atexit.register(shutil.rmtree, TMP_PROFILE, ignore_errors=True)

def get_Driver():
    if DRIVER is None:
        raise RuntimeError("driver_init has not run in this process")
    return DRIVER