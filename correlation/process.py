import multiprocessing
import random


def worker(k, q):
    t = 0
    print("processname", k)
    for i in range(10):
        x = random.randint(1, 3)
        t += x
    q.put(t)

def do():
    q = multiprocessing.Queue()
    jobs = []
    for i in range(10):
        p = multiprocessing.Process(target=worker, args=(str(i), q))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    results = [q.get() for j in jobs]
    print(results)


if __name__ == '__main__':
   do()