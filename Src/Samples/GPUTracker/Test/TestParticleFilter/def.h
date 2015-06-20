#ifndef DEF_H
#define DEF_H

#include<iostream>

#define StateValueType float
#define BlockThreadNum 256
#define GridBlockNum(ParticleNum) (ParticleNum+BlockThreadNum-1)/BlockThreadNum
#define GetThreadID(id,ParticleNum) int id=blockDim.x*blockIdx.x+threadIdx.x;if(id>=ParticleNum){return;}

#define STATE_DEF(StateTypeName, StateNum, ...) \
    typedef union {StateValueType data[StateNum];struct{StateValueType __VA_ARGS__;};} StateTypeName;

#define STATE_NUM(StateTypeName) \
    sizeof(StateTypeName)/sizeof(StateValueType)

#define CUDAFREE(pointer) \
    if(pointer!=NULL){cudaFree(pointer);pointer=NULL;}

#endif // DEF_H

