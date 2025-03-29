import pytest
from playwright.sync_api import BrowserContext, Page

from src.ui_simulator.BitUnixPage import BitUnixPage

BASE_URL = "https://www.bitunix.com/"
DEFAULT_TIMEOUT = 3000


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    browser_args = {
        **browser_context_args,
        "base_url": BASE_URL,
        "locale": "en-US",
        # "viewport": {'width': 2560, 'height': 1440} # Monitor
        "viewport": {"width": 1600, "height": 1200}  # Single

    }
    return browser_args


@pytest.fixture(scope="session")
def browser_context(browser, browser_context_args) -> BrowserContext:
    context = browser.new_context(**browser_context_args)
    context.set_default_timeout(DEFAULT_TIMEOUT)
    yield context


@pytest.fixture()
def page(browser_context, request) -> Page:
    page = browser_context.new_page()
    # Add event listeners for debugging
    page.on("console", lambda msg: print(f"Browser console: {msg.text}"))
    page.on("pageerror", lambda err: print(f"Page error: {err}"))
    page.on("requestfailed", lambda request: print(f"Request failed: {request.url}"))
    return page


@pytest.fixture()
def bitunix_page(page):
    bt_page = BitUnixPage(page)
    return bt_page
