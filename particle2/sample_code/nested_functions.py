x = "global x"  # Global scope


def outermost():
    x = "outermost x"  # Global scope

    def outer():
        x = "enclosing enclosing x"  # Enclosing scope

        def inner():
            x = "enclosing x"  # Local scope

            def innermost():
                x = "local x"  # Local scope
                print(x)  # Prints 'local x'

            innermost()
            print(x)  # Prints 'enclosing x'

        inner()
        print(x)  # Prints 'enclosing enclosing x'

    outer()
    print(x)  # Prints 'outermost x'


outermost()
print(x)  # Prints 'global x'
