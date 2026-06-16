BLOCK_SIZES=(128 256 512 1024)

rm -f ncu_report_b*.log

for b in "${BLOCK_SIZES[@]}"; do
    sed -i "s/^#define BLOCK_SIZE.*/#define BLOCK_SIZE $b/" main.cu
    nvcc -O2 -arch=sm_89 -Xptxas -v main.cu -o main

    if [ $? -ne 0 ]; then
        echo "COMPILE ERROR"
        exit 1
    fi

    LOG_FILE="ncu_report_b${b}.log"

    echo "========== (BLOCK_SIZE = $b) ==========" > $LOG_FILE
    ./main >> $LOG_FILE 2>&1
    
    echo "" >> $LOG_FILE
    echo "========== Nsight Compute 效能報告 ==========" >> $LOG_FILE
    ncu --set basic ./main >> $LOG_FILE 2>&1
    
    echo "DONE"
done