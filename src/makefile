CXXFLAGS=--std=c++11 `pkg-config --cflags --libs opencv` -I../include -lm -static -ffast-math -march=native

ifeq ($(PROFILE),1)
    CXXFLAGS += -pg
endif

ifeq ($(DEBUG),1)
    CXXFLAGS += -g
else
    CXXFLAGS += -O3 
endif

ifeq ($(OMP),1)
    CXXFLAGS += -fopenmp
endif

all: clean libdpm.a
clean: 
	rm -f *.o *.a
libdpm.a: dpm_model.o dpm_feature.o dpm_cascade.o dpm_cascade_detector.o dpm_convolution.o dpm_nms.o
	ar rvs $@ *.o
	
