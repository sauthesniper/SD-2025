def counting_sort_by_digit(arr, digit_place):
    n = len(arr)
    output = [0] * n
    count = [0] * 64  # Contor pentru cifrele de la 0 la 63

    for num in arr:
        digit = (num // digit_place) % 64
        count[digit] += 1

    for i in range(1, 64):
        count[i] += count[i - 1]

    # Construim vectorul sortat in mod stabil
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // digit_place) % 64
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1

    for i in range(n):
        arr[i] = output[i]

def radix_sort(arr):
    if not arr:
        return []

    # Determinam cel mai mare numar pentru a sti cate cifre (baza 64) are
    max_num = max(arr)

    # Aplicam counting sort pentru fiecare pozitie (unitati, 64, 64^2, etc.)
    digit_place = 1
    while max_num // digit_place > 0:
        counting_sort_by_digit(arr, digit_place)
        digit_place *= 64

    return arr
