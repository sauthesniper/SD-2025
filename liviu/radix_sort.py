def counting_sort_by_digit(arr, digit_place):
    n = len(arr)
    output = [0] * n
    count = [0] * 10  # Contor pentru cifrele de la 0 la 9

    for num in arr:
        digit = (num // digit_place) % 10
        count[digit] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    # Construim vectorul sortat 
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // digit_place) % 10
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1

    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    if not arr:
        return []

    # Cautam cel mai mare numar pentru a sti cate cifre are
    max_num = max(arr)

    # Aplicam counting sort pentru fiecare cifra (unitati, zeci, sute etc.)
    digit_place = 1
    while max_num // digit_place > 0:
        counting_sort_by_digit(arr, digit_place)
        digit_place *= 10

    return arr
# arr = [170, 45, 75, 90, 802, 24, 2, 66]
# print("Original:", arr)
# sorted_arr = radix_sort(arr)
# print("Sortat:", sorted_arr)
