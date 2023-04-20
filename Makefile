OBJS=kmeans.o
EXE=example1 example2
CC=gcc-12

CFLAGS=-fopenmp -g -O0

all: $(EXE)

clean:
	@rm -f *.o $(EXE)

example1: $(OBJS) example1.o
	$(CC) $(CFLAGS) $^ -o $@

example2: $(OBJS) example2.o
	$(CC) $(CFLAGS) $^ -o $@