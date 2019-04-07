from flask import session, redirect
from functools import wraps


def check(route):
    @wraps(route)
    def wrapper(*args, **kwargs):
        if "flag" not in session:
            return redirect('/index')
        return route(*args, **kwargs)
    return wrapper
