def fizz_buzz(n: int):
    """Print numbers from 1 to n with FizzBuzz rules."""
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)


# Call the function to run FizzBuzz up to 100
fizz_buzz(100)
