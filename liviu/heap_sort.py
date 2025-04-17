def heapify(arr, n, i):
    largest = i      # presupunem ca radacina este cea mai mare
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[left] > arr[largest]:
        largest = left

    if right < n and arr[right] > arr[largest]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]  # swap
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # Construim heap-ul (max heap)
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # Scoatem elementele din heap unul cate unul
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # mutam radacina la final
        heapify(arr, i, 0)

    return arr

# arr = [12, 11, 13, 5, 6, 7]
# print("Original:", arr)
# sorted_arr = heap_sort(arr)
# print("Sortat:", sorted_arr)
