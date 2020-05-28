import sys
import threading
import time
import random

class MyThread(threading.Thread):
    def __init__(self, id, value):
        threading.Thread.__init__(self)
        self.id = id
        self.value = value

    def run(self):
        time.sleep(1)
        print("Hello, I'm Thread ", self.id, "and I received the number ", self.value)

def main():
    threads = []
    P = int(sys.stdin.readline().strip("\n"))

    for i in range(0, P):
        threads.append(MyThread(i, random.randint(0, 100)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
