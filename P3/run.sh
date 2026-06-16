rm -f run_vlmul_*.log
make clean > /dev/null 2>&1

VLMULS=("m1" "m2" "m4" "m8")

for v in "${VLMULS[@]}"; do
    sed -i "s/^#define VLMUL.*/#define VLMUL \"$v\"/" main.cpp

    LOG_FILE="run_vlmul_${v}.log"

    make > $LOG_FILE 2>&1

    if [ $? -ne 0 ]; then
        echo "MAKE ERROR"
        exit 1
    fi

    
    echo "" >> $LOG_FILE
    echo "========================================" >> $LOG_FILE
    echo " KEY DATA" >> $LOG_FILE
    echo "========================================" >> $LOG_FILE
    grep -E "simSeconds|simTicks|numCycles|simInsts|simOps|hostSeconds|overallMisses|overallMissRate" m5out/stats.txt >> $LOG_FILE

    grep -E "numCycles|simInsts|overallMissRate::total" m5out/stats.txt

    
    make clean > /dev/null 2>&1
done