NVCCFLAGS	:=
NVCC		:= nvcc


all: cypher

cypher: cypher.cu util.h
	$(NVCC) -o cypher cypher.cu $(NVCCFLAGS)


clean:
	rm -rf cypher particle
