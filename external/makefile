nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib = /usr/local/cuda-8.0/lib64
tensorflow = /home/zhangyuge/anaconda3/envs/py2tf/lib/python2.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1
TF_CFLAGS = -I/home/zhangyuge/anaconda3/envs/py2tf/lib/python2.7/site-packages/tensorflow/include -D_GLIBCXX_USE_CXX11_ABI=1
TF_LFLAGS = -L/home/zhangyuge/anaconda3/envs/py2tf/lib/python2.7/site-packages/tensorflow -ltensorflow_framework -I/home/zhangyuge/anaconda3/envs/py2tf/lib/python2.7/site-packages/tensorflow/include/external/nsync/public

all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++-4.8 -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	$(nvcc) -ccbin=g++-4.8 -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++-4.8 -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -lcudart -L $(cudalib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	$(nvcc) -ccbin=g++-4.8 -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2


clean:
	rm tf_approxmatch_so.so
	rm tf_nndistance_so.so
	rm  *.cu.o 
