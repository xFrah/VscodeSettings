import time
from line_profiler import LineProfiler


def cock():
    for i in range(10):
        print(i)
        time.sleep(1)


LineProfiler(cock)
