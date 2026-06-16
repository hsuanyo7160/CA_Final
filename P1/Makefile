GEM5_DIR := /root/gem5
GEM5 := $(GEM5_DIR)/build/RISCV/gem5.opt
CONFIG := $(GEM5_DIR)/configs/deprecated/example/se.py

CC := riscv64-linux-gnu-g++
CFLAGS := -static -O2

SRC := main.cpp
BIN := main

all: run

$(BIN): $(SRC)
	$(CC) $(CFLAGS) $< -o $@

run: $(BIN)
	$(GEM5) $(CONFIG) -c $(CURDIR)/$(BIN) --cpu-type=TimingSimpleCPU --mem-size=256MB --caches --l1i_size=32kB --l1i_assoc=8 --l1d_size=32kB --l1d_assoc=8 --cacheline=32
	grep -E "simSeconds|simTicks|numCycles|simInsts|simOps|hostSeconds|overallMisses|overallMissRate" m5out/stats.txt

clean:
	rm -f $(BIN)
	rm -rf m5out

.PHONY: all run clean
