import os
from typing import Literal

import dotenv
from playwright.sync_api import Page

from settings import logger

dotenv.load_dotenv()


class BitUnixPage:
    BASE_URL = "https://www.bitunix.com/"

    def __init__(self, page: Page) -> None:
        self.browser_page = page
        self.password = os.environ.get("BITUNIX_PSWRD")
        self.email = "karination.z@icloud.com"

    def load(self, url) -> None:
        logger.info(f"Loading {url}")
        self.browser_page.goto(url)

    def go_to_login_page(self):
        self.browser_page.goto(self.BASE_URL + "login")

    def enter_email(self):
        locator = self.browser_page.get_by_role("textbox", name="Enter email address")
        locator.fill(self.email)
        return locator

    def enter_password(self):
        locator = self.browser_page.get_by_role("textbox", name="Please Enter Your Password")
        locator.fill(self.password)
        return locator

    def login(self):
        self.go_to_login_page()
        # email_locator = self.enter_email()
        password_locator = self.enter_password()
        #
        # login_btn = self.browser_page.locator(".auth-base-button-light").get_by_text("Log in")
        # login_btn.click()

    def futures_btn_click(self):
        self.browser_page.get_by_role("link", name="Futures", exact=True).click()

    def open_list_of_coins(self):
        self.browser_page.locator(".kline-title").locator(".bit-icon").click()

    def search_coin(self, coin_name):
        self.browser_page.get_by_role("textbox", name="Search").fill(coin_name)
        self.browser_page.locator("span").filter(has_text=coin_name).last.click()

    @property
    def trading_view_frame(self):
        return self.browser_page.frame_locator("iframe[name^='tradingview_']").first

    def open_lines_tool_menu(self):
        self.trading_view_frame.get_by_role("button", name="Trend line tools").click()

    def select_horizontal_line(self):
        self.trading_view_frame.get_by_text("Horizontal Line⌥ H").click()
        self.trading_view_frame.get_by_label("Chart for ").click()

    @property
    def floating_lines_toolbar_frame(self):
        return self.browser_page.frame_locator("iframe[name^='tradingview_']").first

    def open_floating_lines_toolbar_settings(self):
        self.floating_lines_toolbar_frame.locator('[data-name="settings"]').click()

    def select_color(self, color: Literal["blue", "orange"]):
        self.floating_lines_toolbar_frame.locator('[data-name="color-with-thickness-select"]').click()
        if color == "blue":
            self.floating_lines_toolbar_frame.locator('[style="color: rgb(49, 121, 245);"]').click()
        else:
            self.floating_lines_toolbar_frame.locator('[style="color: rgb(255, 152, 0);"]').click()
        self.floating_lines_toolbar_frame.get_by_role("tab", name="Style").click()

    def select_line_type(self, line_type: Literal["solid", "dashed"]):
        self.floating_lines_toolbar_frame.locator('[data-name="line-style-select"]').click()

        if line_type == "solid":
            self.floating_lines_toolbar_frame.get_by_role("option", name="Line", exact=True).click()
        else:
            self.floating_lines_toolbar_frame.get_by_role("option", name="Dashed line").click()

    def select_coordinates(self, price):
        self.floating_lines_toolbar_frame.get_by_role("tab", name="Coordinates").click()
        self.floating_lines_toolbar_frame.get_by_role("textbox").clear()
        self.floating_lines_toolbar_frame.get_by_role("textbox").fill(str(price))
        self.floating_lines_toolbar_frame.get_by_role("button", name="Ok").click()
