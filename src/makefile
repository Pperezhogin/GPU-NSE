FLOW = flow

arch = -arch=sm_20

CUDA_COMPILE = nvcc $(arch) -c -O3
CUDA_COMPILE_DEVICE = nvcc $(arch) -dc
CUDA_LINK    = nvcc $(arch) -O3

OBJ = device.o kernels.o data.o main.o

$(FLOW): $(OBJ)
	$(CUDA_LINK) $(OBJ) -o $(FLOW)
 
device.o: device.cu device.cuh
	$(CUDA_COMPILE_DEVICE) device.cu
 
kernels.o: kernels.cu kernels.h
	$(CUDA_COMPILE_DEVICE) kernels.cu

data.o: data.cu data.h
	$(CUDA_COMPILE) data.cu

main.o: main.cu defines.h model.h
	$(CUDA_COMPILE) main.cu

clear:
	rm -f $(OBJ) $(FLOW)
	clear
	
profile: $(FLOW)
	nvprof ./$(FLOW)
	
profile1: $(FLOW)
	nvprof --metrics all ./$(FLOW)
