NVCC := nvcc
FILE ?= main
SRC := $(FILE).cu
TARGET := ./output/$(basename $(SRC))
NVCCFLAGS := -O3

all: $(TARGET)

output:
	mkdir -p ./output

$(TARGET): $(SRC) help.cu | output
	$(NVCC) $(NVCCFLAGS) -o $@ $(SRC)
	

run: $(TARGET)
	./$(TARGET)

./output/%: %.cu help.cu | output
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f ./output/$(basename $(SRC)) *.o
	rmdir ./output 2>/dev/null || true

.PHONY: all run clean output
 
