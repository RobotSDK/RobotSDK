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

#include"particlefilterdef.h"
#include"randomgenerator.h"
#include"egotransform.h"

#include<vector>

template<class StateType, class SampleStateType, class RandomStateType>
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
    virtual void initialize(StateType & initialState, SampleStateType & randomOffset)=0;
    __host__ __device__
    virtual void randomnize(RandomStateType & randomOffset)=0;
    __host__ __device__
    virtual void transform(EgoTransform & transform)=0;
    __host__ __device__
    virtual void update(int & deltaMsec)=0;
};

template<class StateType, class SampleStateType, class ParticleType>
__global__
void kernelInitialParticles(StateType objectState, SampleStateType * randomOffset, ParticleType * newParticles, int particleNum)
{
    GetThreadID(id, particleNum);
    ParticleType particle;
    particle.weight=1.0/particleNum;
    particle.initialize(objectState,randomOffset[id]);
    newParticles[id]=particle;
}

template<class RandomStateType, class ParticleType>
__global__
void kernelRandomnizeParticles(RandomStateType * randomOffset, ParticleType * intputParticles, ParticleType * outputParticles, int particleNum)
{
    GetThreadID(id, particleNum);
    ParticleType particle=intputParticles[id];
    particle.randomnize(randomOffset[id]);
    outputParticles[id]=particle;
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

template<class ParticleType>
__global__
void kernelTransformParticles(EgoTransform transform, ParticleType * particles, int totalParticleNum)
{
    GetThreadID(id, totalParticleNum);
    ParticleType particle=particles[id];
    particle.transform(transform);
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

template<class StateType, class SampleStateType, class RandomStateType, class ParticleType, class MeasureDataType, class ParticleMeasureType>
class ParticleFilterBase
{
protected:
    int statenum=STATE_NUM(StateType);
    int particlenum=0;
    int objectsnum=0;
    int totalparticlenum=0;
protected:
    int * d_randomseeds=NULL;
    RandomOffsetGenerator<SampleStateType> * d_samplegenerator=NULL;
    RandomOffsetGenerator<RandomStateType> * d_randomgenerator=NULL;
    SampleStateType * d_samplemin=NULL, * d_samplemax=NULL;
    SampleStateType * d_sampleoffset=NULL;
    RandomStateType * d_randommean=NULL , * d_randomsigma=NULL;
    RandomStateType * d_randomoffset=NULL;
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
    void initialParticleFilter(int particleNum, SampleStateType sampleMin, SampleStateType sampleMax, RandomStateType randomMean, RandomStateType randomSigma)
    {                
        clear();

        particlenum=particleNum;
        objectsnum=0;
        totalparticlenum=0;

        cudaMalloc((void **)(&d_samplegenerator),particlenum*sizeof(RandomOffsetGenerator<SampleStateType>));
        cudaMalloc((void **)(&d_randomgenerator),particlenum*sizeof(RandomOffsetGenerator<RandomStateType>));

        int blocknum=GridBlockNum(particlenum);
        int threadnum=BlockThreadNum;

        generateRandomSeeds();
        kernelSetRandomSeeds<SampleStateType><<<blocknum,threadnum>>>(d_samplegenerator,d_randomseeds,particlenum);
        generateRandomSeeds();
        kernelSetRandomSeeds<RandomStateType><<<blocknum,threadnum>>>(d_randomgenerator,d_randomseeds,particlenum);

        cudaMalloc((void **)(&d_samplemin),sizeof(SampleStateType));
        cudaMemcpy((void *)d_samplemin,&sampleMin,sizeof(SampleStateType),cudaMemcpyHostToDevice);
        cudaMalloc((void **)(&d_samplemax),sizeof(SampleStateType));
        cudaMemcpy((void *)d_samplemax,&sampleMax,sizeof(SampleStateType),cudaMemcpyHostToDevice);
        cudaMalloc((void **)(&d_sampleoffset),particlenum*sizeof(SampleStateType));

        cudaMalloc((void **)(&d_randommean),sizeof(RandomStateType));
        cudaMemcpy((void *)d_randommean,&randomMean,sizeof(RandomStateType),cudaMemcpyHostToDevice);
        cudaMalloc((void **)(&d_randomsigma),sizeof(RandomStateType));
        cudaMemcpy((void *)d_randomsigma,&randomSigma,sizeof(RandomStateType),cudaMemcpyHostToDevice);
        cudaMalloc((void **)(&d_randomoffset),particlenum*sizeof(RandomStateType));

        h_particles=new ParticleType[particlenum];
        h_resample=new float[particlenum];

        return;
    }
    void clear()
    {
        CUDAFREE(d_randomseeds);
        CUDAFREE(d_samplegenerator)
        CUDAFREE(d_randomgenerator);
        CUDAFREE(d_samplemin);CUDAFREE(d_samplemax);
        CUDAFREE(d_sampleoffset);
        CUDAFREE(d_randommean);CUDAFREE(d_randomsigma);
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
    void generateRandomSeeds()
    {
        int * randomseeds=new int[particlenum];
        thrust::generate(randomseeds,randomseeds+particlenum,rand);
        if(d_randomseeds==NULL)
        {
            cudaMalloc((void **)(&d_randomseeds),particlenum*sizeof(int));
        }
        cudaMemcpy(d_randomseeds,randomseeds,sizeof(int)*particlenum,cudaMemcpyHostToDevice);
        delete []randomseeds;
    }
public:
    void addObjectState(const int objectID, const StateType objectState)
    {
        int blocknum=GridBlockNum(particlenum);
        int threadnum=BlockThreadNum;
        kernelGenerateUniformRandomOffset<SampleStateType><<<blocknum,threadnum>>>(d_samplegenerator,d_sampleoffset,d_samplemin,d_samplemax,particlenum);

        ParticleType * d_newparticles;
        cudaMalloc((void **)(&d_newparticles),sizeof(ParticleType)*particlenum);
        kernelInitialParticles<StateType,SampleStateType,ParticleType><<<blocknum,threadnum>>>(objectState,d_sampleoffset,d_newparticles,particlenum);

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
    void addObjectState(const std::vector<int> & objectID, const std::vector<StateType> & objectState)
    {
        int i,n=objectID.size(),m=objectState.size();
        if(n*m==0)
        {
            return;
        }
        int blocknum=GridBlockNum(particlenum);
        int threadnum=BlockThreadNum;
        for(i=0;i<n&&i<m;i++)
        {
            kernelGenerateUniformRandomOffset<SampleStateType><<<blocknum,threadnum>>>(d_samplegenerator,d_sampleoffset,d_samplemin,d_samplemax,particlenum);

            ParticleType * d_newparticles;
            cudaMalloc((void **)(&d_newparticles),sizeof(ParticleType)*particlenum);
            kernelInitialParticles<StateType,SampleStateType,ParticleType><<<blocknum,threadnum>>>(objectState[i],d_sampleoffset,d_newparticles,particlenum);

            d_objects.push_back(d_newparticles);
            objectsid.push_back(objectID[i]);
        }
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
    bool estimateObjectState(int objectID, StateType & objectState, StateValueType & normalizer)
    {
        for(int i=0;i<objectsnum;i++)
        {
            if(objectsid[i]==objectID)
            {
                cudaMemcpy(h_particles,d_objects[i],sizeof(ParticleType)*particlenum,cudaMemcpyDeviceToHost);
                ParticleType particle;
                for(int j=0;j<STATE_NUM(StateType);j++)
                {
                    normalizer=0;
                    for(int k=0;k<particlenum;k++)
                    {
                        particle.state.data[j]+=h_particles[k].state.data[j]*h_particles[k].weight;
                        normalizer+=h_particles[k].weight;
                    }
                }
                objectState=particle.state;
                return 1;
            }
        }
        return 0;
    }
    std::vector<int> estimateObjectState(std::vector<StateType> & objectsState, std::vector<StateValueType> & normalizer)
    {
        objectsState.resize(objectsnum);
        normalizer.resize(objectsnum);
        for(int i=0;i<objectsnum;i++)
        {
            cudaMemcpy(h_particles,d_objects[i],sizeof(ParticleType)*particlenum,cudaMemcpyDeviceToHost);
            ParticleType particle;
            for(int j=0;j<STATE_NUM(StateType);j++)
            {
                normalizer[i]=0;
                for(int k=0;k<particlenum;k++)
                {
                    particle.state.data[j]+=h_particles[k].state.data[j]*h_particles[k].weight;
                    normalizer[i]+=h_particles[k].weight;
                }
            }
            objectsState[i]=particle.state;
        }
        return objectsid;
    }
    bool removeObject(int objectID)
    {
        for(int i=0;i<objectsnum;i++)
        {
            if(objectsid[i]==objectID)
            {
                CUDAFREE(d_objects[i]);
                d_objects.erase(d_objects.begin()+i);
                objectsid.erase(objectsid.begin()+i);
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

                return 1;
            }
        }
        return 0;
    }
    void removeObject(std::vector<int> & objectID)
    {
        for(int j=0;j<objectID.size();j++)
        {
            for(int i=0;i<objectsnum;i++)
            {
                if(objectsid[i]==objectID[j])
                {
                    CUDAFREE(d_objects[i]);
                    d_objects.erase(d_objects.begin()+i);
                    objectsid.erase(objectsid.begin()+i);
                    break;
                }
            }
        }
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
            kernelGenerateNormalRandomOffset<RandomStateType><<<blocknum,threadnum>>>(d_randomgenerator,d_randomoffset,d_randommean,d_randomsigma,particlenum);
            kernelRandomnizeParticles<RandomStateType,ParticleType><<<blocknum,threadnum>>>(d_randomoffset,d_objects[i],d_particles+i*particlenum,particlenum);
        }
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
    void removeParticles(float & minWeightThreshold)
    {
        if(objectsnum==0)
        {
            return;
        }
        for(int i=objectsnum-1;i>=0;i--)
        {
            if(h_maxweight[i]<=minWeightThreshold)
            {
                CUDAFREE(d_objects[i]);
                d_objects.erase(d_objects.begin()+i);
                objectsid.erase(objectsid.begin()+i);
            }
        }
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
public:
    void debugParticleFilterGetParticles(std::vector< std::vector<StateType> > & particleState, std::vector< std::vector<float> > & particleWeight)
    {
        if(objectsnum==0)
        {
            return;
        }
        particleState.resize(objectsnum);
        particleWeight.resize(objectsnum);
        for(int i=0;i<objectsnum;i++)
        {
            particleState[i].resize(particlenum);
            particleWeight[i].resize(particlenum);
            cudaMemcpy(h_particles,d_particles+i*particlenum,sizeof(ParticleType)*particlenum,cudaMemcpyDeviceToHost);
            for(int j=0;j<particlenum;j++)
            {
                particleState[i][j]=h_particles[j].state;
                particleWeight[i][j]=h_particles[j].weight;
            }
        }
    }
    void debugParticleFilterSetParticles(std::vector< std::vector<StateType> > & particleState, std::vector< std::vector<float> > & particleWeight)
    {
        if(objectsnum!=particleState.size())
        {
            return;
        }
        for(int i=0;i<objectsnum;i++)
        {
            if(particlenum!=particleState[i].size())
            {
                return;
            }
            for(int j=0;j<particlenum;j++)
            {
                h_particles[j].state=particleState[i][j];
                h_particles[j].weight=particleWeight[i][j];
            }
            cudaMemcpy(d_particles+i*particlenum,h_particles,sizeof(ParticleType)*particlenum,cudaMemcpyHostToDevice);
        }
    }
};

#endif // PARTICLEFILTERBASE_H

