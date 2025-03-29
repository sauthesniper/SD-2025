import random
import time

def merge(v, st, mid, dr):
    st_n = mid - st + 1
    dr_n = dr - mid

    S = [v[i] for i in range(st, mid + 1)]
    D = [v[i] for i in range(mid + 1, dr + 1)]

    i,j,fi = 0, 0, st

    while i < st_n and j < dr_n:
        if S[i] <= D[j]:
            v[fi] = S[i]
            i += 1
        else:
            v[fi] = D[j]
            j += 1
        fi += 1
    
    while i < st_n:
        v[fi] = S[i]
        fi += 1
        i += 1

    while j < dr_n:
        v[fi] = D[j]
        fi += 1
        j += 1
    

def merge_sort(v, st, dr):
    if st >= dr: return

    mid = (st + dr) // 2
    merge_sort(v, st, mid)
    merge_sort(v, mid + 1, dr)
    merge(v, st, mid, dr)


print("Generating random array...")
v = [random.randint(1, 2_000_000) for i in range(10_000_000)]
print("Random array has been generated\n")

print("Starting sort...")
start = time.time()
merge_sort(v, 0, len(v) - 1)
end = time.time()
print(f"Sorting ended with an elapsed time of {end - start}")