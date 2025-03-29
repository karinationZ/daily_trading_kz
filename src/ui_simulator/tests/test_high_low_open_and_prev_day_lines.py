import csv

from settings import PROJECT_DIR

CSV_FILE = PROJECT_DIR + f"/artifacts/filtered_futures/filtered_futures.csv"


def read_csv_to_dict():
    data = {}

    with open(CSV_FILE, "r") as file:
        reader = csv.DictReader(file, delimiter=",")
        for row in reader:
            if not row["candidate"]:
                continue
            pair = row["asset"]
            prev_day_high = float(row["prev_day_high"])
            prev_day_low = float(row["prev_day_low"])
            first_five_min_high = float(row["first_five_min_high"])
            first_five_min_low = float(row["first_five_min_low"])

            data[pair] = [
                ("blue", "solid", prev_day_high),
                ("orange", "solid", prev_day_low),
                ("blue", "dashed", first_five_min_high),
                ("orange", "dashed", first_five_min_low)
            ]

    return data


def test(bitunix_page):
    data = read_csv_to_dict()
    logged_in = False
    for coin, pair_info in data.items():
        page = bitunix_page

        if not logged_in:
            page.login()
            page.browser_page.wait_for_timeout(60000)
            logged_in = True

        curr_coin = None
        for color, line_type, price in pair_info:
            if curr_coin is None:
                page.futures_btn_click()
                page.open_list_of_coins()
                page.search_coin(coin)
                curr_coin = coin

            page.open_lines_tool_menu()
            page.select_horizontal_line()
            page.open_floating_lines_toolbar_settings()
            page.select_color(color=color)
            page.select_line_type(line_type="solid")
            page.select_coordinates(price=price)

    page.browser_page.wait_for_event("close", timeout=0)
