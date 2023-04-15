
def check_arg_absence(func):
    def wrapper(*args, **kwargs):
        for elem in kwargs.values():
            if elem is None:
                raise ValueError(f"Argument '{f'{elem=}'.split('=')[0]}' cannot be None "
                                 f"for calling function '{func.__name__}'")
        return func(*args, **kwargs)
    return wrapper
