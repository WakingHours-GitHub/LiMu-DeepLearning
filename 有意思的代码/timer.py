import time


"""
看到d2l源码中, 一个有意思的计时器, 这里复现一下
"""
class Timer:
    def __init__(self) -> None:
        self.times = []
        # self.tik = self.start() # start time, when create object.
    
    def start(self) -> float:
        self.tik = time.start()
    
    def stop(self) -> float:
        self.times.append(time.time() - self.tik)
        return self.times[-1] # return time, that now add


    
