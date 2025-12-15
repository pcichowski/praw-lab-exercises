CC = gcc
CFLAGS = -O3 -march=native -fopenmp
LDFLAGS = -lm

TARGET = sobel_omp
SRC = sobel_omp.c main.c

all:
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
