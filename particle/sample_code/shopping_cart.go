package main

import (
	"fmt"
)

// formatPrice formats the amount as a string with two decimal places.
func formatPrice(amount float64) string {
	return fmt.Sprintf("$%.2f", amount)
}

// ShoppingCart represents a shopping cart with a list of item prices.
type ShoppingCart struct {
	items []float64
}

// NewShoppingCart creates a new ShoppingCart instance.
func NewShoppingCart() *ShoppingCart {
	return &ShoppingCart{items: []float64{}}
}

// AddItem adds a price to the shopping cart and calculates the total.
func (cart *ShoppingCart) AddItem(price float64) {
	cart.items = append(cart.items, price)
	cart.CalculateTotal()
}

// CalculateTotal calculates the total price of items in the cart.
func (cart *ShoppingCart) CalculateTotal() string {
	total := 0.0
	for _, price := range cart.items {
		total += price
	}
	return formatPrice(total)
}

// processShoppingCart creates a new shopping cart, adds an item, and returns the cart.
func processShoppingCart() *ShoppingCart {
	cart := NewShoppingCart()
	cart.AddItem(10.99)
	return cart
}

func main() {
	cart := processShoppingCart()
	fmt.Println("Total:", cart.CalculateTotal())
}