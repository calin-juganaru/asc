from threading import *

numThreads = 4

self.numThreads = numThreads
self.countThreads = numThreads
self.countLock = Lock()
self.countLock2 = Lock()
self.threadsSem = Semaphore(0)
self.threadsSem2 = Semaphore(numThreads)
self.countThreads2 = numThreads

def barrier(self):
	self.threadsSem2.acquire()
	self.countLock.acquire()
	self.countThreads -= 1

	if (countThreads == 0):
		for i in range(self.numThreads):
			self.ThreadsSem.release()
		countThreads = numThreads

	self.countLock.release()
	self.ThreadsSem.acquire()

	self.countLock2.acquire()
	countThreads2 -= 1	# se traduce prin mai multe instr de asamblare:
				# mov eax, [countThreads2]
				# sub eax, 1
				# [mov countThreads2], eax
	# poate aparea switch de context in cadrul instructiunii efectiv (printre cele de sus atomice^)
   if (countThreads2 == 0):
		for i in range(numThreads):
			self.ThreadsSem2.release()
		countThreads2 = numThreads
		self.countLock2.release()