def foo(bar, baz=None, *_args, **kw_args):
    pass


def qux(bar: int, baz: int = None, *_args, **kw_args):
    pass


foo(7, baz=10, qux=11, quux=12)
