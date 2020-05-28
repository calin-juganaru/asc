from threading import enumerate, Event, Thread, Condition, Lock
import random

from concurrent.futures import ThreadPoolExecutor
SubStr = "ACTGCC"

def randomAdn():
    return "".join(random.choice('ATGC') for _ in range(1, 10000))

def findSubString(index):
    return SubStr in ADN_LIST[index]

if __name__ == "__main__":
    random.seed(SubStr)
    ADN_LIST = [randomAdn() for _ in range(0, 100)]
    with ThreadPoolExecutor(max_workers=30) as executor:
        results = executor.map(findSubString, range(0, 100))
    for result in results:
        print(result)
