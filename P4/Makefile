CXX = g++
CXXFLAGS = -O2
TARGET = main_cpp
SRC = main_cpu.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)