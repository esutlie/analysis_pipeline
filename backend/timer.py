import time


class Timer:
    def __init__(self):
        self.last_time = time.time()
        self.i = 0

    def tic(self, label=''):
        new_time = time.time()
        print(f'{self.i}: {new_time - self.last_time:.8f} {label}')
        self.last_time = new_time
        self.i += 1
