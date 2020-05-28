from threading import *

class SimpleBarrier():
    def __init__(self, num_threads):
        self.num_threads = num_threads
        self.count_threads = self.num_threads    # contorizeaza numarul de thread-uri ramase
        self.count_lock = Lock()                 # protejeaza accesarea/modificarea contorului
        self.threads_sem = Semaphore(0)          # blocheaza thread-urile ajunse

    def wait(self):
        with self.count_lock:
            self.count_threads -= 1
            if self.count_threads == 0:          # a ajuns la bariera si ultimul thread
                for i in range(self.num_threads):
                    self.threads_sem.release()   # incrementarea semaforului va debloca num_threads thread-uri
        self.threads_sem.acquire()               # num_threads-1 threaduri se blocheaza aici
                                                 # contorul semaforului se decrementeaza de num_threads ori

class MyThread(Thread):
    def __init__(self, tid, barrier):
        Thread.__init__(self)
        self.tid = tid
        self.barrier = barrier

    def run(self):
        print ("I'm Thread " + str(self.tid) + " before\n")
        self.barrier.wait()
        print ("I'm Thread " + str(self.tid) + " after barrier\n")