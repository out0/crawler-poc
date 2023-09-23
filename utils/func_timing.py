import time

def time_exec(func: callable, name: str) -> any:
    exec_start = time.time()
    ret = func()
    exec_finish = time.time()
    print(f"{name} execution time: {(exec_finish - exec_start)*1000} ms")
    return ret