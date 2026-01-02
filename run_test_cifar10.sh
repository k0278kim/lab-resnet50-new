#!/bin/bash

echo "=== test_cifar10.py cusin=1 model=1 파일 실행 시작 ==="
python test_cifar10.py --cusin 1 --model 1

echo "=== test_cifar10.py cusin=2 model=1 파일 실행 시작 ==="
python test_cifar10.py --cusin 2 --model 1

echo "=== test_cifar10.py cusin=3 model=1 파일 실행 시작 ==="
python test_cifar10.py --cusin 3 --model 1

echo "=== test_cifar10.py cusin=4 model=1 파일 실행 시작 ==="
python test_cifar10.py --cusin 4 --model 1

echo "=== test_cifar10.py cusin=1 model=2 파일 실행 시작 ==="
python test_cifar10.py --cusin 1 --model 2

echo "=== test_cifar10.py cusin=2 model=2 파일 실행 시작 ==="
python test_cifar10.py --cusin 2 --model 2

echo "=== test_cifar10.py cusin=3 model=2 파일 실행 시작 ==="
python test_cifar10.py --cusin 3 --model 2

echo "=== test_cifar10.py cusin=4 model=2 파일 실행 시작 ==="
python test_cifar10.py --cusin 4 --model 2