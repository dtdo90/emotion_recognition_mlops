import time
from functools import wraps

def timing(f):
    """ Decorator for  timing
        Use: @timing
    """
    @wraps(f)
    def wrapper(*args,**kwargs):
        start=time.time()
        result=f(*args,**kwargs)
        end=time.time()
        print("%r took %2.5f seconds" % (f.__name__,end-start))
        return result
    
    return wrapper