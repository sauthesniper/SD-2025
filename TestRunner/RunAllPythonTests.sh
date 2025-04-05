#!/bin/bash

directory="$1"

files=("$directory"/*)

python3 ./PythonSorts/RunAllTests.py "${files[@]}"
