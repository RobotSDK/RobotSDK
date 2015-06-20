#ifndef PARTICLEFILTERBASE_H
#define PARTICLEFILTERBASE_H

#include<cuda.h>
#include<cuda_runtime.h>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<thrust/transform.h>
#include<thrust/reduce.h>
#include<thrust/adjacent_difference.h>
#include<thrust/sequence.h>
#include<thrust/execution_policy.h>

#include"def.h"
#include"randomgenerator.h"
#include"egotransform.h"

#include<vector>

template<class StateType>
class ParticleBase
{
public:
    float weight;
    StateType state;
public:
    __host__ __device__
    ParticleBase()
    {
        weight=0;
        for(int i=0;i<STATE_NUM(StateType);i++)
        {
            state.data[i]=0;
        }
    }
public:
    __host__ __device__
    virtual void initialize(StateType & initialState, StateType & randomOffset)=0;
    __host__ __device__
    virtual void randomnize(StateType & randomOffset)=0;
    __host__ __device__
    virtual void transform(EgoTransform & transform)=0;
    __host__ __device__
    virtual void update(int & deltaMsec)=0;
};

template<class StateType, class ParticleType>
__global__
void kernelInitialParticles(StateType objectState, StateType * randomOffset, ParticleType * newParticles, int particleNum)
{
    GetThreadID(id, particleNum);
    ParticleType particle;
    particle.weight=1.0/particleNum;
    particle.initialize(objectState,randomOffset[id]);
    newParticles[id]=particle;
}

template<class StateType, class ParticleType>
__global__
void kernelRandomnizeParticles(StateType * randomOffset, ParticleType * intputParticles, ParticleType * outputParticles, int particleNum)
{
    GetThreadID(id, particleNum);
    ParticleType particle=intputParticles[id];
    particle.randomnize(randomOffset[id]);
    outputParticles[id]=particle;
}

template<class ParticleType>
__global__
void kernelTransformParticles(EgoTransform transform, ParticleType * particles, int totalParticleNum)
{
    GetThreadID(id, totalParticleNum);
    ParticleType particle=particles[id];
    particle.transform(transform);
    particles[id]=particle;
}

template<class ParticleType>
__global__
void kernelUpdateParticles(int deltaMsec, ParticleType * particles, int totalParticleNum)
{
    GetThreadID(id, totalParticleNum);
    ParticleType particle=particles[id];
    particle.update(deltaMsec);
    particles[id]=particle;
}

template<class ParticleType, class MeasureDataType, class ParticleMeasureType>
__global__
void kernelMeasureParticles(ParticleType * particles, MeasureDataType measureData, int totalParticleNum)
{
    GetThreadID(id, totalParticleNum);
    ParticleMeasureType particlemeasure;
    particles[id].weight=particlemeasure.particleMeasure(particles[id].state,measureData);
}

template<class ParticleType>
__global__
void kernelResampleParticles(float * resample, float * weightSum, ParticleType * cumulative, ParticleType * particles, int particleNum, int totalParticleNum)
{
    GetThreadID(id, totalParticleNum);
    int objectid=id/particleNum;
    if(weightSum[objectid]<=0)
    {
        return;
    }
    int startid=objectid*particleNum;
    int endid=startid+particleNum;
    for(int i=startid;i<endid;i++)
    {
        if(resample[id]<=cumulative[i].weight/weightSum[objectid])
        {
            particles[id]=cumulative[i];
            particles[id].weight=1.0f/particleNum;
            return;
        }
    }
    particles[id]=cumulative[endid-1];
    particles[id].weight=1.0f/particleNum;
}

template<class StateType, class MeasureDataType>
class ParticleMeasureBase
{
public:
    __host__ __device__
    virtual float particleMeasure(StateType & state, MeasureDataType & measureData)=0;
};

template<class StateType, class ParticleType, class MeasureDataType, class ParticleMeasureType>
class ParticleFilterBase
{
protected:
    int statenum=STATE_NUM(StateType);
    int particlenum=0;
    int objectsnum=0;
    int totalparticlenum=0;
protected:
    int * d_randomseeds=NULL;
    RandomOffsetGenerator<StateType> * d_randomgenerator=NULL;
    StateType * d_statemin=NULL, * d_statemax=NULL;
    StateType * d_statemean=NULL , * d_statesigma=NULL;
    StateType * d_randomoffset=NULL;
protected:
    thrust::minstd_rand rng;
    thrust::random::uniform_real_distribution<float> dist;
    float * h_weightsum=NULL;
    float * d_weightsum=NULL;
    float * h_maxweight=NULL;
    float * h_resample=NULL;
    float * d_resample=NULL;
protected:
    std::vector<ParticleType *> d_objects;
    std::vector<int> objectsid;
    ParticleType * d_particles=NULL;    //totalparticlenum
    ParticleType * d_cumulative=NULL;
    ParticleType * h_particles=NULL;    //particlenum
public:
    ParticleFilterBase()
    {
        rng=thrust::minstd_rand(time(NULL));
        dist=thrust::random::uniform_real_distribution<StateValueType>(0,1.0);
        d_objects.clear();
    }
    virtual ~ParticleFilterBase()
    {
        clear();
    }
public:
    void initialParticleFilter(int particleNum, StateType stateMin, StateType stateMax, StateType stateMean, StateType stateSigma)
    {                
        clear();

        particlenum=particleNum;
        objectsnum=0;
        totalparticlenum=0;

        cudaMalloc((void **)(&d_randomseeds),particlenum*sizeof(int));
        generateRandomSeeds(d_randomseeds);

        cudaMalloc((void **)(&d_randomgenerator),particlenum*sizeof(RandomOffsetGenerator<StateType>));

        int blocknum=GridBlockNum(particlenum);
        int threadnum=BlockThreadNum;
        kernelSetRandomSeeds<StateType><<<blocknum,threadnum>>>(d_randomgenerator,d_randomseeds,particlenum);

        cudaMalloc((void **)(&d_statemin),sizeof(StateType));
        cudaMemcpy((void *)d_statemin,&stateMin,sizeof(StateType),cudaMemcpyHostToDevice);
        cudaMalloc((void **)(&d_statemax),sizeof(StateType));
        cudaMemcpy((void *)d_statemax,&stateMax,sizeof(StateType),cudaMemcpyHostToDevice);
        cudaMalloc((void **)(&d_statemean),sizeof(StateType));
        cudaMemcpy((void *)d_statemean,&stateMean,sizeof(StateType),cudaMemcpyHostToDevice);
        cudaMalloc((void **)(&d_statesigma),sizeof(StateType));
        cudaMemcpy((void *)d_statesigma,&stateSigma,sizeof(StateType),cudaMemcpyHostToDevice);
        cudaMalloc((void **)(&d_randomoffset),particlenum*sizeof(StateType));

        h_particles=new ParticleType[particlenum];
        h_resample=new float[particlenum];

        return;
    }
    void clear()
    {
        CUDAFREE(d_randomseeds);
        CUDAFREE(d_randomgenerator);
        CUDAFREE(d_statemin);CUDAFREE(d_statemax);
        CUDAFREE(d_statemean);CUDAFREE(d_statesigma);
        CUDAFREE(d_randomoffset);
        CUDAFREE(d_weightsum);
        CUDAFREE(d_resample);
        for(int i=0;i<d_objects.size();i++)
        {
            CUDAFREE(d_objects[i]);
        }
        d_objects.clear();
        objectsid.clear();
        objectsnum=0;

        CUDAFREE(d_particles);
        CUDAFREE(d_cumulative);
        totalparticlenum=0;

        if(h_particles!=NULL)
        {
            delete []h_particles;
            h_particles=NULL;
        }
        if(h_weightsum!=NULL)
        {
            delete []h_weightsum;
            h_weightsum=NULL;
        }
        if(h_maxweight!=NULL)
        {
            delete []h_maxweight;
            h_maxweight=NULL;
        }
        if(h_resample!=NULL)
        {
            delete []h_resample;
            h_resample=NULL;
        }
    }
protected:
    void generateRandomSeeds(int * d_randomSeeds)
    {
        int * randomseeds=new int[particlenum];
        thrust::generate(randomseeds,randomseeds+particlenum,rand);
        cudaMemcpy(d_randomseeds,randomseeds,sizeof(int)*particlenum,cudaMemcpyHostToDevice);
    }
public:
    void addObjectState(int objectID, StateType objectState)
    {
        int blocknum=GridBlockNum(particlenum);
        int threadnum=BlockThreadNum;
        kernelGenerateUniformRandomOffset<StateType><<<blocknum,threadnum>>>(d_randomgenerator,d_randomoffset,d_statemin,d_statemax,particlenum);

        ParticleType * d_newparticles;
        cudaMalloc((void **)(&d_newparticles),sizeof(ParticleType)*particlenum);
        kernelInitialParticles<StateType,ParticleType><<<blocknum,threadnum>>>(objectState,d_randomoffset,d_newparticles,particlenum);

        d_objects.push_back(d_newparticles);
        objectsid.push_back(objectID);
        objectsnum=objectsid.size();
        totalparticlenum=objectsnum*particlenum;

        CUDAFREE(d_particles);
        CUDAFREE(d_cumulative);
        CUDAFREE(d_weightsum);
        CUDAFREE(d_resample);
        cudaMalloc((void **)(&d_particles),sizeof(ParticleType)*totalparticlenum);
        cudaMalloc((void **)(&d_cumulative),sizeof(ParticleType)*totalparticlenum);
        cudaMalloc((void **)(&d_weightsum),sizeof(float)*objectsnum);
        cudaMalloc((void **)(&d_resample),sizeof(float)*totalparticlenum);

        if(h_weightsum!=NULL)
        {
            delete []h_weightsum;
            h_weightsum=NULL;
        }
        h_weightsum=new float[objectsnum];
        if(h_maxweight!=NULL)
        {
            delete []h_maxweight;
            h_maxweight=NULL;
        }
        h_maxweight=new float[objectsnum];
    }
    bool getObjectState(int objectID, StateType & objectState)
    {
        for(int i=0;i<objectsnum;i++)
        {
            if(objectsid[i]==objectID)
            {
                cudaMemcpy(h_particles,d_objects[i],sizeof(ParticleType)*particlenum,cudaMemcpyDeviceToHost);
                ParticleType particle;
                for(int j=0;j<STATE_NUM(StateType);j++)
                {
                    for(int k=0;k<particlenum;k++)
                    {
                        particle.state.data[j]+=h_particles[k].state.data[j];
                    }
                    particle.state.data[j]/=particlenum;
                }
                objectState=particle.state;
                return 1;
            }
        }
        return 0;
    }
    std::vector<int> getObjectState(std::vector<StateType> & objectsState)
    {
        objectsState.resize(objectsnum);
        for(int i=0;i<objectsnum;i++)
        {
            cudaMemcpy(h_particles,d_objects[i],sizeof(ParticleType)*particlenum,cudaMemcpyDeviceToHost);
            ParticleType particle;
            for(int j=0;j<STATE_NUM(StateType);j++)
            {
                for(int k=0;k<particlenum;k++)
                {
                    particle.state.data[j]+=h_particles[k].state.data[j];
                }
                particle.state.data[j]/=particlenum;
            }
            objectsState[i]=particle.state;
        }
        return objectsid;
    }
public:
    void randomnizeParticles()
    {
        if(objectsnum==0)
        {
            return;
        }
        int blocknum=GridBlockNum(particlenum);
        int threadnum=BlockThreadNum;
        for(int i=0;i<objectsnum;i++)
        {
            kernelGenerateNormalRandomOffset<StateType><<<blocknum,threadnum>>>(d_randomgenerator,d_randomoffset,d_statemean,d_statesigma,particlenum);
            kernelRandomnizeParticles<StateType,ParticleType><<<blocknum,threadnum>>>(d_randomoffset,d_objects[i],d_particles+i*particlenum,particlenum);
        }
    }
    void transformParticles(EgoTransform & transform)
    {
        if(objectsnum==0)
        {
            return;
        }
        int blocknum=GridBlockNum(totalparticlenum);
        int threadnum=BlockThreadNum;
        kernelTransformParticles<ParticleType><<<blocknum,threadnum>>>(transform,d_particles,totalparticlenum);
    }
    void updateParticles(int & deltaMsec)
    {
        if(objectsnum==0)
        {
            return;
        }
        int blocknum=GridBlockNum(totalparticlenum);
        int threadnum=BlockThreadNum;
        kernelUpdateParticles<ParticleType><<<blocknum,threadnum>>>(deltaMsec,d_particles,totalparticlenum);
    }
    virtual void measureParticles(MeasureDataType & measureData)
    {
        int blocknum=GridBlockNum(totalparticlenum);
        int threadnum=BlockThreadNum;
        kernelMeasureParticles<ParticleType,MeasureDataType,ParticleMeasureType><<<blocknum,threadnum>>>(d_particles,measureData,totalparticlenum);
    }
    void resampleParticles()
    {
        if(objectsnum==0)
        {
            return;
        }
        for(int i=0;i<objectsnum;i++)
        {
            cudaMemcpy(h_particles,d_particles+i*particlenum,sizeof(ParticleType)*particlenum,cudaMemcpyDeviceToHost);
            h_maxweight[i]=h_particles[0].weight;
            for(int j=1;j<particlenum;j++)
            {
                h_particles[j].weight+=h_particles[j-1].weight;
                h_resample[j]=dist(rng);
                if(h_maxweight[i]<h_particles[j].weight)
                {
                    h_maxweight[i]=h_particles[j].weight;
                }
            }
            h_weightsum[i]=h_particles[particlenum-1].weight;
            cudaMemcpy(d_cumulative+i*particlenum,h_particles,sizeof(ParticleType)*particlenum,cudaMemcpyHostToDevice);
            cudaMemcpy(d_resample+i*particlenum,h_resample,sizeof(float)*particlenum,cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_weightsum,h_weightsum,sizeof(float)*objectsnum,cudaMemcpyHostToDevice);

        int blocknum=GridBlockNum(totalparticlenum);
        int threadnum=BlockThreadNum;
        kernelResampleParticles<ParticleType><<<blocknum,threadnum>>>(d_resample,d_weightsum,d_cumulative,d_particles,particlenum,totalparticlenum);

        for(int i=0;i<objectsnum;i++)
        {
            cudaMemcpy(d_objects[i],d_particles+i*particlenum,sizeof(ParticleType)*particlenum,cudaMemcpyDeviceToDevice);
        }
    }
public:
    void removieParticles(float & maxWeightThreshold)
    {
        if(objectsnum==0)
        {
            return;
        }
        for(int i=0;i<objectsnum;i++)
        {
            if(h_maxweight[i]<maxWeightThreshold)
            {
                //delete this object;
            }
        }
    }
};

#endif // PARTICLEFILTERBASE_H

