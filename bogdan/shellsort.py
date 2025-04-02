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
    gap=len(arr)
    while gap!=1 :
        gap//=2
        for i in range(len(arr)-gap) :
            if arr[i]>arr[i+gap] :
                arr[i], arr[i+gap] = arr[i+gap], arr[i]
    return arr

filename = "numbers.txt"
rand(5, filename)
arr = r(filename)


start = time.time()
sorted_arr = shell_sort(arr)
print(sorted_arr)
end = time.time()
print(f"Time taken: {end - start:.6f} seconds")
