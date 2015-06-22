#ifndef PARTICLEFILTERDEF_H
#define PARTICLEFILTERDEF_H

#include<vector>

#define StateValueType float
#define BlockThreadNum 256
#define GridBlockNum(ParticleNum) (ParticleNum+BlockThreadNum-1)/BlockThreadNum
#define GetThreadID(id,ParticleNum) int id=blockDim.x*blockIdx.x+threadIdx.x;if(id>=ParticleNum){return;}
#define CUDAFREE(pointer) if(pointer!=NULL){cudaFree(pointer);pointer=NULL;}

#define STATE_TYPE(PFName) PFName##_State
#define STATE_DEF(PFName, StateNum, ...) \
    typedef union {StateValueType data[StateNum];struct{StateValueType __VA_ARGS__;};} STATE_TYPE(PFName);
#define STATE_NUM(StateTypeName) \
    sizeof(StateTypeName)/sizeof(StateValueType)

#define MEASUREDATA_TYPE(PFName) PFName##_MeasureData
#define MEASUREDATA_DEF(PFName, MeasureDataType) \
    typedef MeasureDataType MEASUREDATA_TYPE(PFName);

#define PF_DEF(PFName) \
    typedef ParticleBase<STATE_TYPE(PFName)> PFName##_ParticleBase; \
    class PFName##_Particle : public PFName##_ParticleBase \
    { \
    public: \
        __host__ __device__ \
        void initialize(STATE_TYPE(PFName) & initialState, STATE_TYPE(PFName) & randomOffset); \
        __host__ __device__ \
        void randomnize(STATE_TYPE(PFName) & randomOffset); \
        __host__ __device__ \
        void transform(EgoTransform & transform); \
        __host__ __device__ \
        void update(int & deltaMsec); \
    }; \
    typedef ParticleMeasureBase<STATE_TYPE(PFName),MEASUREDATA_TYPE(PFName)> PFName##_MeasureBase; \
    class PFName##_Measure : public PFName##_MeasureBase \
    { \
    public: \
        __host__ __device__ \
        float particleMeasure(STATE_TYPE(PFName) & state, MEASUREDATA_TYPE(PFName) & measureData); \
    }; \
    typedef ParticleFilterBase<STATE_TYPE(PFName),PFName##_Particle,MEASUREDATA_TYPE(PFName),PFName##_Measure> PFName##_Base; \
    class PFName : public PFName##_Base \
    { \
    public: \
        void measureParticles(MEASUREDATA_TYPE(PFName) & measureData); \
    };

#define PARTICLE_INITIALIZE_FUNC(PFName, initialStateArgName, randomOffsetArgName) \
    __host__ __device__ void PFName##_Particle::initialize(STATE_TYPE(PFName) & initialStateArgName, STATE_TYPE(PFName) & randomOffsetArgName)

#define PARTICLE_RANDOMNIZE_FUNC(PFName, randomOffsetArgName) \
    __host__ __device__ void PFName##_Particle::randomnize(STATE_TYPE(PFName) & randomOffsetArgName)

#define PARTICLE_UPDATE_FUNC(PFName, deltaMsecArgName) \
    __host__ __device__ void PFName##_Particle::update(int & deltaMsecArgName)

#define PARTICLE_TRANSFORM_FUNC(PFName, transformArgName) \
    __host__ __device__ void PFName##_Particle::transform(EgoTransform & transformArgName)

#define PARTICLE_MEASURE_FUNC(PFName, particleStateArgName, measureDataArgName) \
    __host__ __device__ float PFName##_Measure::particleMeasure(STATE_TYPE(PFName) & particleStateArgName, MEASUREDATA_TYPE(PFName) & measureDataArgName)

#define PARTICLE_FILTER_MEASURE_FUNC(PFName, measureDataArgName) \
    void PFName::measureParticles(MEASUREDATA_TYPE(PFName) & measureDataArgName)

#define PARTICLE_FILTER_MEASURE_DEFAULT_FUNC(PFName) \
    PARTICLE_FILTER_MEASURE_FUNC(PFName, measureData) \
    { \
        PFName##_Base::measureParticles(measureData); \
    }

#define PARTICLE_FILTER_INTERACT_FUNCS_DECL(PFName) \
    extern "C" void PF_##PFName##_initialParticleFilter(int particleNum, STATE_TYPE(PFName) & stateMin, STATE_TYPE(PFName) & stateMax, STATE_TYPE(PFName) & stateMean, STATE_TYPE(PFName) & stateSigma); \
    extern "C" void PF_##PFName##_addObjectState(int objectID, STATE_TYPE(PFName) & objectState); \
    extern "C" void PF_##PFName##_advanceParticleFilter(int deltaMsec, MEASUREDATA_TYPE(PFName) & measureData); \
    extern "C" void PF_##PFName##_advanceParticleFilter2D(int deltaMsec, MEASUREDATA_TYPE(PFName) & measureData, StateValueType x1=0, StateValueType y1=0, StateValueType theta1=0, StateValueType x2=0, StateValueType y2=0, StateValueType theta2=0); \
    extern "C" void PF_##PFName##_advanceParticleFilter3D(int deltaMsec, MEASUREDATA_TYPE(PFName) & measureData, StateValueType * transformMatrix=NULL); \
    extern "C" void PF_##PFName##_removeParticles(float minWeightThreshold); \
    extern "C" bool PF_##PFName##_estimateObject(int objectID, STATE_TYPE(PFName) & objectState); \
    extern "C" void PF_##PFName##_estimateObjects(std::vector<int> & objectsID, std::vector<STATE_TYPE(PFName)> & objectsState);

#define PARTICLE_INSTANCE(PFName) PFName##_particlefilter
#define PARTICLE_FILTER_INTERACT_FUNCS(PFName) \
    PFName PARTICLE_INSTANCE(PFName); \
    extern "C" void PF_##PFName##_initialParticleFilter(int particleNum, STATE_TYPE(PFName) & stateMin, STATE_TYPE(PFName) & stateMax, STATE_TYPE(PFName) & stateMean, STATE_TYPE(PFName) & stateSigma) \
    { \
        PARTICLE_INSTANCE(PFName).initialParticleFilter(particleNum,stateMin,stateMax,stateMean,stateSigma); \
    } \
    extern "C" void PF_##PFName##_addObjectState(int objectID, STATE_TYPE(PFName) & objectState) \
    { \
        PARTICLE_INSTANCE(PFName).addObjectState(objectID,objectState); \
    } \
    extern "C" void PF_##PFName##_advanceParticleFilter(int deltaMsec, MEASUREDATA_TYPE(PFName) & measureData) \
    { \
        PARTICLE_INSTANCE(PFName).randomnizeParticles(); \
        PARTICLE_INSTANCE(PFName).updateParticles(deltaMsec); \
        PARTICLE_INSTANCE(PFName).measureParticles(measureData); \
        PARTICLE_INSTANCE(PFName).resampleParticles(); \
        cudaDeviceSynchronize(); \
    } \
    extern "C" void PF_##PFName##_advanceParticleFilter2D(int deltaMsec, MEASUREDATA_TYPE(PFName) & measureData, StateValueType x1, StateValueType y1, StateValueType theta1, StateValueType x2, StateValueType y2, StateValueType theta2) \
    { \
        EgoTransform transform(x1,y1,theta1,x2,y2,theta2); \
        PARTICLE_INSTANCE(PFName).randomnizeParticles(); \
        PARTICLE_INSTANCE(PFName).updateParticles(deltaMsec); \
        PARTICLE_INSTANCE(PFName).transformParticles(transform); \
        PARTICLE_INSTANCE(PFName).measureParticles(measureData); \
        PARTICLE_INSTANCE(PFName).resampleParticles(); \
        cudaDeviceSynchronize(); \
    } \
    extern "C" void PF_##PFName##_advanceParticleFilter3D(int deltaMsec, MEASUREDATA_TYPE(PFName) & measureData, StateValueType * transformMatrix) \
    { \
        PARTICLE_INSTANCE(PFName).randomnizeParticles(); \
        PARTICLE_INSTANCE(PFName).updateParticles(deltaMsec); \
        if(transformMatrix!=NULL) \
        { \
            EgoTransform transform(transformMatrix); \
            PARTICLE_INSTANCE(PFName).transformParticles(transform); \
        } \
        PARTICLE_INSTANCE(PFName).measureParticles(measureData); \
        PARTICLE_INSTANCE(PFName).resampleParticles(); \
        cudaDeviceSynchronize(); \
    } \
    extern "C" void PF_##PFName##_removeParticles(float minWeightThreshold) \
    { \
        PARTICLE_INSTANCE(PFName).removeParticles(minWeightThreshold); \
    } \
    extern "C" bool PF_##PFName##_estimateObject(int objectID, STATE_TYPE(PFName) & objectState) \
    { \
        return PARTICLE_INSTANCE(PFName).estimateObjectState(objectID,objectState); \
    } \
    extern "C" void PF_##PFName##_estimateObjects(std::vector<int> & objectsID, std::vector<STATE_TYPE(PFName)> & objectsState) \
    { \
        objectsID=PARTICLE_INSTANCE(PFName).estimateObjectState(objectsState); \
    }


#endif // PARTICLEFILTERDEF_H

