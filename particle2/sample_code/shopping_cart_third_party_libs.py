import numpy as np
import requests


def format_price(amount):
    return f"${amount:.2f}"


class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, price):
        self.items.append(price)
        self.calculate_total()

    def calculate_total(self):
        # Using numpy to calculate the total
        total = np.sum(self.items)
        return format_price(total)

    def fetch_exchange_rate(self, currency_code):
        # Using requests to fetch exchange rate from a mock API
        response = requests.get(f"https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        return data["rates"].get(currency_code, None)


def process_shopping_cart():
    cart = ShoppingCart()
    cart.add_item(10.99)
    # Example of using the fetch_exchange_rate method
    exchange_rate = cart.fetch_exchange_rate("EUR")
    print(f"Exchange rate for EUR: {exchange_rate}")
    return cart
