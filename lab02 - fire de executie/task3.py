"""

Consumer 65 consumed espresso
Factory 7 produced a nice small espresso
Consumer 87 consumed cappuccino
Factory 9 produced an italian medium cappuccino
Consumer 90 consumed americano
Consumer 84 consumed espresso
Factory 8 produced a strong medium americano
Consumer 135 consumed cappuccino
Consumer 94 consumed americano
"""
import sys
from threading import Thread, Semaphore, Lock
import random

lock = Lock()
results = []
sizes = ["small", "medium", "large"]

""" =========================================================================================== """

class Coffee:
    """ Base class """
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def get_name(self):
        """ Returns the coffee name """
        return self.name

    def get_size(self):
        """ Returns the coffee size """
        return self.size


class Espresso(Coffee):
    """ Espresso implementation """
    def __init__(self, size):
        Coffee.__init__(self, "espresso", size)

    def get_message(self):
        """ Output message """
        return "strong {} {}".format(self.size, self.name)

class Cappuccino(Coffee):
    """ Cappuccino implementation """
    def __init__(self, size):
        Coffee.__init__(self, "cappuccino", size)

    def get_message(self):
        """ Output message """
        return "italian {} {}".format(self.size, self.name)

class Americano(Coffee):
    """ Espresso implementation """
    def __init__(self, size):
        Coffee.__init__(self, "americano", size)

    def get_message(self):
        """ Output message """
        return "nice {} {}".format(self.size, self.name)

coffees = [Espresso, Americano, Cappuccino]

""" =========================================================================================== """

class Distributor:
    def __init__(self, size):
        self.size = size
        self.mutexP = Lock()
        self.mutexC = Lock()
        self.empty = Semaphore(size)
        self.full = Semaphore(0)
        self.buffer = [None] * size
        self.last = 0
        self.first = 0

    def put(self, elem):
        self.empty.acquire()

        with self.mutexP:
            self.buffer[self.last] = elem
            self.last = (self.last + 1) % self.size

        self.full.release()

    def get(self):
        self.full.acquire()
        elem = None

        with self.mutexC:
            elem = self.buffer[self.first]
            self.first = (self.first + 1) % self.size

        self.empty.release()

        return elem

""" =========================================================================================== """

class User(Thread):
    def __init__(self, id, buffer):
        Thread.__init__(self)
        self.id = id
        self.buffer = buffer

    def run(self):
        while True:
            coffee = self.buffer.get()
            print("Consumer {} consumed {}".format(self.id, coffee.get_message()))

""" =========================================================================================== """

class CoffeeFactory(Thread):
    def __init__(self, id, buffer):
        Thread.__init__(self)
        self.id = id
        self.buffer = buffer

    def run(self):
        global sizes
        global coffees
        for i in range(0, 100):
            coffee_type = random.choice(coffees)
            coffee_size = random.choice(sizes)
            coffee = coffee_type(coffee_size)
            print("Factory {} produced {}".format(self.id, coffee.get_message()))
            self.buffer.put(coffee)

""" =========================================================================================== """

def main():
    N = int(sys.stdin.readline().strip("\n"))
    P = int(sys.stdin.readline().strip("\n"))
    C = int(sys.stdin.readline().strip("\n"))

    buffer = Distributor(N)
    threads = [None] * (P + C)
    results = range(0, N)

    for i in range(0, P):
        threads[i] = CoffeeFactory(i, buffer)

    for i in range(0, C):
        threads[i + P] = User(i, buffer)

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()
