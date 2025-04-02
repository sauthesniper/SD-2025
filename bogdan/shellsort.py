import random
import time

def rand(n, filename):
    with open(filename, 'w') as file:
        file.write(f"{n}\n")
        numbers = [str(random.randint(1, 10)) for _ in range(n)]
        file.write(" ".join(numbers) + "\n")

def r(filename):
    with open(filename, 'r') as file:
        file.readline()
        return list(map(int, file.readline().split()))

def shell_sort(arr):
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2 
    return arr


filename = "numbers.txt"
rand(5, filename)
arr = r(filename)


start = time.time()
print(arr)
sorted_arr = shell_sort(arr)
print(sorted_arr)
end = time.time()
print(f"Time taken: {end - start:.6f} seconds")
