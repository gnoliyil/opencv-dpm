COMMON_FLAGS=--std=c++11 `pkg-config --cflags --libs opencv` -I../include -lm 

ifeq ($(PROFILE),1)
    COMMON_FLAGS += -pg
endif

ifeq ($(DEBUG),1)
    COMMON_FLAGS += -g
    NVCCFLAGS += -G
else
    COMMON_FLAGS += -O3 
endif

ifeq ($(OMP),1)
    NVCCFLAGS += -Xcompiler -fopenmp
    CXXFLAGS += -fopenmp
endif

CXXFLAGS += $(COMMON_FLAGS) -static -ffast-math -march=native
NVCCFLAGS += $(COMMON_FLAGS) --use_fast_math

all: clean libdpm.a

clean: 
	rm -f *.o *.a

libdpm.a: dpm_model.cpp.o dpm_feature.cpp.o dpm_cascade.cpp.o dpm_cascade_detector.cpp.o \
          dpm_convolution.cpp.o dpm_nms.cpp.o dpm_feature_gpu.cu.o 
	ar rvs $@ *.o
	
%.cu.o: %.cu
	nvcc -c $< -o $@ $(NVCCFLAGS) 

%.cpp.o: %.cpp
	g++ -c $< -o $@ $(CXXFLAGS)
