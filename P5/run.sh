PATTERNS=(1 2 4 8 16)
rm -f ncu_report_p*.log

for p in "${PATTERNS[@]}"; do
    echo " NUM_PATTERNS = $p"

    sed -i "s/^#define NUM_PATTERNS.*/#define NUM_PATTERNS $p/" main.cu

    nvcc -O2 -arch=sm_89 -Xptxas -v main.cu -o main

    if [ $? -ne 0 ]; then
        echo "COMPILE ERROR"
        exit 1
    fi

    ./main >> $LOG_FILE 2>&1

    LOG_FILE="ncu_report_p${p}.log"

    ncu --set basic ./main >> $LOG_FILE 2>&1
    
    echo "DONE!"
done