mkdir -p Log
rm -f Log/ncu_report_b*.log Log/cpu_report.log

make clean > /dev/null 2>&1
make > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "CPU COMPILE ERROR"
    exit 1
fi

echo "========== CPU Result ==========" > Log/cpu_report.log
./main_cpp >> Log/cpu_report.log 2>&1
echo ""

BLOCK_SIZES=(128 256 512 1024)

for b in "${BLOCK_SIZES[@]}"; do
    sed -i "s/^#define BLOCK_SIZE.*/#define BLOCK_SIZE $b/" main.cu

    nvcc -O2 -arch=sm_89 -Xptxas -v main.cu -o main

    if [ $? -ne 0 ]; then
        echo "GPU COMPILE ERROR"
        exit 1
    fi

    LOG_FILE="Log/ncu_report_b${b}.log"

    echo "========== GPU Result (BLOCK_SIZE = $b) ==========" > $LOG_FILE
    ./main >> $LOG_FILE 2>&1
    
    echo "" >> $LOG_FILE
    echo "========== Nsight Compute  ==========" >> $LOG_FILE
    ncu --set basic ./main >> $LOG_FILE 2>&1
    
    echo "DONE"
done