
import functools

def FilterFactory(f):
    def wrapper(*args, **kwargs):
        @functools.wraps(f)
        def call(field, context=None):
            return f(field, *args, context=context, **kwargs)
        return call
    return wrapper
