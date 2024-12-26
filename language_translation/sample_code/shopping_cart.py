def format_price(amount):
    return f"${amount:.2f}"


class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, price):
        self.items.append(price)
        self.calculate_total()

    def calculate_total(self):
        total = sum(self.items)
        return format_price(total)


def process_shopping_cart():
    cart = ShoppingCart()
    cart.add_item(10.99)
    return cart
