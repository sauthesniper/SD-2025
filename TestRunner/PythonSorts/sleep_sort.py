import asyncio
import random
import time

def rand(n, filename):
    with open(filename, 'w') as file:
        file.write(f"{n}\n")
        numbers = [str(random.randint(1, 10)) for _ in range(n)]
        file.write(" ".join(numbers) + "\n")

def read(filename):
    with open(filename, 'r') as file:
        file.readline() 
        return list(map(int, file.readline().split()))

async def sleepProcess(num, output, scale_factor):
    await asyncio.sleep(num * scale_factor)
    output.append(num)

async def sleep_sort(sequence):
    output = []
    max_val = max(sequence, default=1)
    scale_factor = 1 / (max_val * len(sequence) * 10) 
    tasks = [sleepProcess(num, output, scale_factor) for num in sequence]
    await asyncio.gather(*tasks)
    return output

#async def main():
#    filename = "numbers.txt"
#    rand(100000, filename)
#    start_time = time.time()
#    sequence = read(filename)
#    sorted_sequence = await sleep_sort(sequence)
#    end_time = time.time()
#    print(f"Time taken: {end_time - start_time:.6f} seconds")

#asyncio.run(main())
