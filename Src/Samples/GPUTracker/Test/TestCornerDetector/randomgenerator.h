#ifndef RANDOMGENERATOR_H
#define RANDOMGENERATOR_H

#include<cuda.h>
#include<cuda_runtime.h>
#include<thrust/random/linear_congruential_engine.h>
#include<thrust/random/uniform_real_distribution.h>
#include<thrust/random/normal_distribution.h>

#include"particlefilterdef.h"

template<class StateType>
class RandomOffsetGenerator
{
public:
    thrust::minstd_rand rng;
public:
    __host__ __device__
    void setSeed(int & seed)
    {
        rng=thrust::minstd_rand(seed);
    }
public:
    __host__ __device__
    void generateUniformRandomOffset(StateType & randomOffset, StateType * stateMin, StateType * stateMax)
    {
        for(int i=0;i<STATE_NUM(StateType);i++)
        {
            randomOffset.data[i]=thrust::random::uniform_real_distribution<StateValueType>(stateMin->data[i],stateMax->data[i])(rng);
        }
    }
    __host__ __device__
    void generateNormalRandomOffset(StateType & randomOffset, StateType * stateMean, StateType * stateSigma)
    {
        for(int i=0;i<STATE_NUM(StateType);i++)
        {
            if(stateSigma->data[i]>0)
            {
                randomOffset.data[i]=thrust::random::normal_distribution<StateValueType>(stateMean->data[i],stateSigma->data[i])(rng);
            }
            else
            {
                randomOffset.data[i]=0;
            }
        }
    }
};

template<class StateType>
__global__
void kernelSetRandomSeeds(RandomOffsetGenerator<StateType> * randomGenerator, int * randomSeeds, int particleNum)
{
    GetThreadID(id,particleNum);
    randomGenerator[id].setSeed(randomSeeds[id]);
}

template<class StateType>
__global__
void kernelGenerateUniformRandomOffset(RandomOffsetGenerator<StateType> * randomGenerator, StateType * randomOffset, StateType * stateMin, StateType * stateMax, int particleNum)
{
    GetThreadID(id,particleNum);
    randomGenerator[id].generateUniformRandomOffset(randomOffset[id],stateMin,stateMax);
}

template<class StateType>
__global__
void kernelGenerateNormalRandomOffset(RandomOffsetGenerator<StateType> * randomGenerator, StateType * randomOffset, StateType * stateMean, StateType * stateSigma, int particleNum)
{
    GetThreadID(id,particleNum);
    randomGenerator[id].generateNormalRandomOffset(randomOffset[id],stateMean,stateSigma);
}

#endif // RANDOMGENERATOR_H
