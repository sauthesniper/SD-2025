def nat_merge_sort(arr):
    v = []
    begin = 0
    for i in range(1, len(arr)):
        if arr[i] < arr[i-1]:
            v.append(arr[begin:i])
            begin = i
    v.append(arr[begin:])

    while len(v) > 1:
        v1 = []
        for i in range(0, len(v), 2):
            if i + 1 < len(v):
                merged = merge(v[i], v[i+1])
                v1.append(merged)
            else:
                v1.append(v[i])
        v = v1

    return v[0]

def merge(left, right):
    v = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            v.append(left[i])
            i += 1
        else:
            v.append(right[j])
            j += 1
    v.extend(left[i:])
    v.extend(right[j:])
    return v