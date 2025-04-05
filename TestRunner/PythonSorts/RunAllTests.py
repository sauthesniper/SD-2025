import time
import asyncio
import os
import sys
from radix_sort import radix_sort
from merge_sort import merge_sort
from shell_sort import shell_sort

from natMerge_sort import nat_merge_sort
from heap_sort import heap_sort

header = "Test file,Radix sort,Merge sort,Shell sort,Heap sort,Natural merge sort"
print(header)


for file_path in sys.argv[1:]:
    with open(file_path, 'r') as file:
        n = int(file.readline().strip())
        arr = [int(x) for x in file.readline().strip().split()]
        sortedArr = sorted(arr)
        arrCopy = arr[:]

        tableRow = f"{os.path.basename(file_path)},"

        arrCopy = arr[:]
        startTime = time.time()
        radix_sort(arrCopy)
        endTime = time.time()
        elapsedTime = (endTime - startTime) * 1000 # in ms
        if arrCopy == sortedArr:
            tableRow += f"{elapsedTime:.0f}ms,"
        else:
            tableRow += "FAIL,"
        
        arrCopy = arr[:]
        startTime = time.time()
        merge_sort(arrCopy)
        endTime = time.time()
        elapsedTime = (endTime - startTime) * 1000 # in ms
        if arrCopy == sortedArr:
            tableRow += f"{elapsedTime:.0f}ms,"
        else:
            tableRow += "FAIL,"

        arrCopy = arr[:]
        startTime = time.time()
        arrCopy = shell_sort(arrCopy)
        endTime = time.time()
        elapsedTime = (endTime - startTime) * 1000 # in ms
        if arrCopy == sortedArr:
            tableRow += f"{elapsedTime:.0f}ms,"
        else:
            tableRow += "FAIL,"

        arrCopy = arr[:]
        startTime = time.time()
        heap_sort(arrCopy)
        endTime = time.time()
        elapsedTime = (endTime - startTime) * 1000 # in ms
        if arrCopy == sortedArr:
            tableRow += f"{elapsedTime:.0f}ms,"
        else:
            tableRow += "FAIL,"

        arrCopy = arr[:]
        startTime = time.time()
        arrCopy = nat_merge_sort(arrCopy)
        endTime = time.time()
        elapsedTime = (endTime - startTime) * 1000 # in ms
        if arrCopy == sortedArr:
            tableRow += f"{elapsedTime:.0f}ms"
        else:
            tableRow += "FAIL"

        print(tableRow)



