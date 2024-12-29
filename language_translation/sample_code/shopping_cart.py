def format_price(amount: float) -> str:
    return f"${amount:.2f}"


class ShoppingCart:
    def __init__(self) -> None:
        self.items: list[float] = []

    def add_item(self, price: float) -> None:
        self.items.append(price)
        self.calculate_total()

    def calculate_total(self) -> str:
        total: float = sum(self.items)
        return format_price(total)


def process_shopping_cart() -> ShoppingCart:
    cart: ShoppingCart = ShoppingCart()
    cart.add_item(10.99)
    return cart
