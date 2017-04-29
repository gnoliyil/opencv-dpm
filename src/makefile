all: libdpm.a

clean: 
	rm *.o *.a

libdpm.a: dpm_model.cpp.o dpm_feature.cpp.o dpm_cascade.cpp.o dpm_cascade_detector.cpp.o \
          dpm_convolution.cpp.o dpm_nms.cpp.o dpm_feature_gpu.cu.o 
	ar rvs $@ *.o
	
%.cu.o: %.cu
	nvcc -c $< -o $@ --std=c++11 `pkg-config --cflags --libs opencv` -I../include -lm -O3 --use_fast_math 

%.cpp.o: %.cpp
	g++ -c $< -o $@ --std=c++11 `pkg-config --cflags --libs opencv` -I../include -lm -O3 -static -ffast-math -march=native
