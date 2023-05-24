OBJS=kmeans.o
EXE=example2

CC=ladcomp -env mpicc

all: $(EXE)

clean:
	@rm -f *.o $(EXE)

example2: $(OBJS) example2.o
	$(CC) $^ -o $@