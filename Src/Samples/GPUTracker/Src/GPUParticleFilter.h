#ifndef GPUPARTICLEFILTER_H
#define GPUPARTICLEFILTER_H

#include<cuda.h>
#include<cuda_runtime.h>
#include<thrust/random/linear_congruential_engine.h>
#include<thrust/random/uniform_real_distribution.h>
#include<thrust/random/normal_distribution.h>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/transform.h>
#include<thrust/reduce.h>
#include<thrust/adjacent_difference.h>
#include<time.h>

#define ADD_STATE(stateNum,stateList) \
    union{struct{stateList};float data[stateNum];}state

class Particle
{
public:
    float weight;
    ADD_STATE(6,float x;float y;float theta;float v;float width;float height;);
};

class GPUParticleFilter
{
public:
    GPUParticleFilter();
};

#endif // GPUPARTICLEFILTER_H
