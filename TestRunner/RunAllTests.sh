#!/bin/bash

directory="$1"
program="$2"

files=("$directory"/*)

# bold="\e[1m"
# reset="\e[0m"
# printf "${bold}%-25s %-15s %-15s %-15s %-15s %-15s${reset}\n" "Name" "Allocation" "GPU Transfer" "Sorting" "CPU Transfer" "Total"

echo "Name,Allocation,GPU Transfer,Sorting,CPU Transfer,Total"

# Run program
for file in "${files[@]}"; do
	output=$("$program" "$file")
	name=$(basename "$file")
	alloc=$(echo "$output" | grep "GPU memory allocation" | awk '{print $NF}' | sed 's/ms//')
	transfer_gpu=$(echo "$output" | grep "Memory transfer to GPU" | awk '{print $NF}' | sed 's/ms//')
	sorting=$(echo "$output" | grep "Sorting on GPU" | awk '{print $NF}' | sed 's/ms//')
	transfer_cpu=$(echo "$output" | grep "Memory transfer to CPU" | awk '{print $NF}' | sed 's/ms//')

	total=$((alloc + transfer_gpu + sorting + transfer_cpu))

	printf "%s,%s,%s,%s,%s,%s\n" "$name" "${alloc}ms" "${transfer_gpu}ms" "${sorting}ms" "${transfer_cpu}ms" "${total}ms"
done
