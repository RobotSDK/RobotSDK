for(sourcefile, SOURCES){
    tmpfind=$$find(sourcefile, .cu)
    isEmpty(tmpfind){
    }
    else{
        CUDA_SOURCES+=$$sourcefile
    }
}

for(sourcefile, DISTFILES){
    tmpfind=$$find(sourcefile, .cu)
    isEmpty(tmpfind){
    }
    else{
        CUDA_SOURCES+=$$sourcefile
    }
}


isEmpty(CUDA_SOURCES){
    message(Please add .cu files to SOURCES variable in .pro file)
}
else{
    message(Successfully get .cu files from SOURCES: $$CUDA_SOURCES)

    CUDA_DIR = /usr/local/cuda
    CUDA_ARCH = sm_35

    INCLUDEPATH  += $$CUDA_DIR/include

    CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
    NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

    cuda.input = CUDA_SOURCES
    cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
    cuda.commands = $$CUDA_DIR/bin/nvcc -std=c++11 -rdc=true --compiler-options '-fPIC' -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                    $$CUDA_INC -L$$CUDA_DIR/lib64 -lcudart -lcuda -lcudadevrt ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                    2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
    cuda.dependency_type = TYPE_C
    cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}

    QMAKE_EXTRA_COMPILERS += cuda
    QMAKE_PRE_LINK = $$CUDA_DIR/bin/nvcc --compiler-options '-fPIC' -arch=$$CUDA_ARCH -dlink $(OBJECTS) -o ${OBJECTS_DIR}cuda_dlink.o -L$$CUDA_DIR/lib64 -lcudart -lcuda -lcudadevrt

    LIBS += ${OBJECTS_DIR}cuda_dlink.o
    LIBS += -L$$CUDA_DIR/lib64 -lcudart
    LIBS += -L$$CUDA_DIR/lib64 -lcuda
    LIBS += -L$$CUDA_DIR/lib64 -lcudadevrt

    SOURCES += $$CUDA_SOURCES
    SOURCES -= $$CUDA_SOURCES
}
