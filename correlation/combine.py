'''
Created on 2020年7月21日

@author: sea
'''
from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from time import sleep


def send_request(req_url, json):
    # print(req_url + "     " + json)
    sleep(3)
    return req_url


if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=3)
    all_task = []
    for i in range(10):
        task = executor.submit(send_request, "url--" + str(i), str(i) + "json")
        all_task.append(task)
        print("all_task size is " + str(len(all_task)))
    for future in as_completed(all_task):
        result = future.result()
        print(result)
