#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H

#include"egotransform.h"
#include"particlefilterbase.h"

STATE_DEF(VehicleState, 6, x,y,theta,v,width,length)

struct TargetVehicle
{
    VehicleState state;
};

typedef ParticleBase<VehicleState> ParticleBaseType;
class VehicleParticle : public ParticleBaseType
{
public:
    __host__ __device__
    void initialize(VehicleState & initialState, VehicleState & randomOffset);
    __host__ __device__
    void randomnize(VehicleState & randomOffset);
    __host__ __device__
    void transform(EgoTransform & transform);
    __host__ __device__
    void update(int & deltaMsec);
};

typedef ParticleMeasureBase<VehicleState,TargetVehicle> ParticleMeasureBaseType;
class VehicleParticleMeasure : public ParticleMeasureBaseType
{
public:
    __host__ __device__
    float particleMeasure(VehicleState &state, TargetVehicle &measureData);
};

typedef ParticleFilterBase<VehicleState,VehicleParticle,TargetVehicle,VehicleParticleMeasure> ParticleFilterBaseType;
class VehicleParticleFilter : public ParticleFilterBaseType
{
public:
    void measureParticles(TargetVehicle & measureData);
};

#endif // PARTICLEFILTER_H

