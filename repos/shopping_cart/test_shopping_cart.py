import unittest

from shopping_cart import ShoppingCart, format_price, CONST_PRICE


class TestShoppingCart(unittest.TestCase):

    def test_format_price(self) -> None:
        self.assertEqual(format_price(10), "$10.00")
        self.assertEqual(format_price(10.5), "$10.50")
        self.assertEqual(format_price(10.555), "$10.55")  # Rounding test

    def test_add_item(self) -> None:
        cart: ShoppingCart = ShoppingCart()
        cart.add_item(CONST_PRICE)
        self.assertEqual(cart.items, [10.99])

    def test_calculate_total(self) -> None:
        cart: ShoppingCart = ShoppingCart()
        cart.add_item(10.99)
        cart.add_item(5.01)
        self.assertEqual(cart.calculate_total(), "$16.00")

    def test_empty_cart_total(self) -> None:
        cart: ShoppingCart = ShoppingCart()
        self.assertEqual(cart.calculate_total(), "$0.00")


if __name__ == "__main__":
    unittest.main()
