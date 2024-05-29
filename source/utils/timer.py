def timer(timer_name=""):
    def inner(func):
        def func_wrapper(*args, **kwargs):
            from time import time
            time_start = time()
            result = func(*args, **kwargs)
            time_end = time()
            time_spend = time_end - time_start
            if time_spend < 60:
                print(f"Task {timer_name} cost time: {time_spend:.3f} s")
            elif time_spend < 3600:
                print(f"Task {timer_name} cost time: {time_spend / 60:.3f} min ({time_spend % 60:.3f} s)")
            else:
                print(f"Task {timer_name} cost time: {time_spend / 3600:.3f} h ({time_spend % 3600 / 60:.3f} min {time_spend % 3600 % 60:.3f} s)")
            return result
        return func_wrapper
    return inner