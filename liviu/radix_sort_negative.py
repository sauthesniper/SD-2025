BASE = 10

def counting_sort_by_digit(arr, digit_place):
    global BASE
    n = len(arr)
    output = [0] * n
    count = [0] * BASE  # Contor pentru cifrele de la 0 la 63

    for num in arr:
        digit = (num // digit_place) % BASE
        count[digit] += 1

    for i in range(1, BASE):
        count[i] += count[i - 1]

    # Construim vectorul sortat in mod stabil
    for i in range(n - 1, -1, -1):
        digit = (arr[i] // digit_place) % BASE
        output[count[digit] - 1] = arr[i]
        count[digit] -= 1

    for i in range(n):
        arr[i] = output[i]

def _radix_sort(arr):
    global BASE
    if not arr:
        return []

    # Determinam cel mai mare numar pentru a sti cate cifre (baza BASE) are
    max_num = max(arr)

    # Aplicam counting sort pentru fiecare pozitie (unitati, BASE, BASE^2, etc.)
    digit_place = 1
    while max_num // digit_place > 0:
        counting_sort_by_digit(arr, digit_place)
        digit_place *= BASE

    return arr

def radix_sort(arr):
    neg = [-x for x in arr if x < 0]
    non_neg = [x for x in arr if x >= 0]

    _radix_sort(neg)
    _radix_sort(non_neg)
    return [-x for x in reversed(neg)] + non_neg


