#!/bin/bash

mkdir -p Log
rm -f Log/*.log

PATTERNS=(1 2 4 8 16)

for p in "${PATTERNS[@]}"; do
    echo "========================================"
    echo " NUM_PATTERNS = $p"
    echo "========================================"

    # ------------------------------------------
    # [CPU]
    # ------------------------------------------
    sed -i "s/^const int NUM_PATTERNS.*/const int NUM_PATTERNS = $p;/" main_cpu.cpp

    make clean > /dev/null 2>&1
    make > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "make error"
        exit 1
    fi

    CPU_LOG="Log/cpu_report_p${p}.log"
    echo "========== CPU Result (NUM_PATTERNS = $p) ==========" > $CPU_LOG
    ./main_cpp >> $CPU_LOG 2>&1


    # ------------------------------------------
    # [GPU]
    # ------------------------------------------
    sed -i "s/^#define NUM_PATTERNS.*/#define NUM_PATTERNS $p/" main.cu
    nvcc -O2 -arch=sm_89 -Xptxas -v main.cu -o main

    if [ $? -ne 0 ]; then
        echo "nvcc error"
        exit 1
    fi

    GPU_LOG="Log/ncu_report_p${p}.log"
    
    echo "========== GPU Result (NUM_PATTERNS = $p) ==========" > $GPU_LOG
    ./main >> $GPU_LOG 2>&1

    echo "" >> $GPU_LOG
    echo "========== Nsight Compute  ==========" >> $GPU_LOG
    ncu --set basic ./main >> $GPU_LOG 2>&1
    
done

echo "DONE!!!"