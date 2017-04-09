all: libdpm.a 
clean: 
	rm *.o
libdpm.a: dpm_model.o dpm_feature.o dpm_cascade.o dpm_cascade_detector.o dpm_convolution.o dpm_nms.o
	ar rvs $@ *.o
	
%.o: %.cpp
	g++ -c $< -o $@ -O2 --std=c++11 `pkg-config --cflags --libs opencv` -I../include -lm -fopenmp -pg

