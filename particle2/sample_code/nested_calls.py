def outer_function():
    def inner_function():
        return "Hello from the inner function!"

    return inner_function


# Call the outer function, which returns the inner function, and then call the inner function
result = outer_function()()
print(result)
