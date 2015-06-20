#include"particlefilter.h"

__host__ __device__
void VehicleParticle::initialize(VehicleState & initialState, VehicleState & randomOffset)
{
    state.x=initialState.x+randomOffset.x;
    state.y=initialState.y+randomOffset.y;
    state.theta=initialState.theta+randomOffset.theta;
    state.v=initialState.v+randomOffset.v;
    state.width=initialState.width+randomOffset.width;
    state.length=initialState.length+randomOffset.length;
    if(state.length<state.width)
    {
        state.length=state.width;
    }
}

__host__ __device__
void VehicleParticle::randomnize(VehicleState &randomOffset)
{
    state.v+=randomOffset.v;
    state.theta+=randomOffset.theta;
}

__host__ __device__
void VehicleParticle::transform(EgoTransform & transform)
{
    transform.transformState2D(state.x,state.y,state.theta);
}

__host__ __device__
void VehicleParticle::update(int & deltaMsec)
{
    float msec=deltaMsec/1000.0;
    float dis=msec*state.v;
    state.x+=dis*cos(state.theta);
    state.y+=dis*sin(state.theta);
}

__host__ __device__
float VehicleParticleMeasure::particleMeasure(VehicleState & state, TargetVehicle & measureData)
{
    float deltax=measureData.state.x-state.x;
    float deltay=measureData.state.y-state.y;
    float deltatheta=measureData.state.theta-state.theta;
    float deltaw=measureData.state.width-state.width;
    float deltal=measureData.state.length-state.length;

    float distance=exp(-(deltax*deltax+deltay*deltay));
    float angle=exp(-(deltatheta*deltatheta));
    float geometry=exp(-(deltaw*deltaw+deltal*deltal));

    return distance*angle*geometry;
}

void VehicleParticleFilter::measureParticles(TargetVehicle & measureData)
{
    ParticleFilterBaseType::measureParticles(measureData);
}

#include<iostream>
#include<chrono>
using namespace std::chrono;

extern "C" void cudaMain()
{
    VehicleParticleFilter particlefilter;
    VehicleState statemin={-1,-1,-0.1,-5,-1,-1};
    VehicleState statemax={1,1,0.1,5,1,1};
    VehicleState statemean={0,0,0.0,0,0};
    VehicleState statesigma={1,1,0.1,5,1,1};
    particlefilter.initialParticleFilter(10000,statemin,statemax,statemean,statesigma);

    for(int j=0;j<20;j++)
    {
        VehicleState state={-1+j*0.1f,1-j*0.1f,0.1f-j*0.01f,1,2,3};
        particlefilter.addObjectState(j,state);
    }

    std::vector<VehicleState> states;
    std::vector<int> objectid;

    EgoTransform transform(0,0,0,0,0,0);
    int msec=1000;
    TargetVehicle target;
    target.state={0,0,0,1.5,2.3,2.8};

    for(int j=0;j<100;j++)
    {
        float dis=target.state.v*msec/1000.0;
        target.state.x+=dis*cos(target.state.theta);
        target.state.y+=dis*sin(target.state.theta);

        std::cout<<"========================================\n";
        std::cout<<"#"<<j<<"\t"<<target.state.x<<"\t"<<target.state.y<<"\t"<<target.state.theta<<"\t"<<target.state.v<<"\t"<<target.state.width<<"\t"<<target.state.length<<"\n";

        milliseconds start,end;
        std::chrono::duration<double> elapsed_seconds;

        start = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
        particlefilter.randomnizeParticles();
        particlefilter.transformParticles(transform);
        particlefilter.updateParticles(msec);
        particlefilter.measureParticles(target);
        particlefilter.resampleParticles();
        end = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
        elapsed_seconds = end-start;
        std::cout<<"Particle Filter:"<<elapsed_seconds.count()<<" s\n";

        start = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
        objectid=particlefilter.getObjectState(states);
        end = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
        elapsed_seconds = end-start;
        std::cout<<"Estimate:"<<elapsed_seconds.count()<<" s\n";
        for(int i=0;i<states.size();i++)
        {
            std::cout<<objectid[i]<<"\t"<<states[i].x<<"\t"<<states[i].y<<"\t"<<states[i].theta<<"\t"<<states[i].v<<"\t"<<states[i].width<<"\t"<<states[i].length<<"\n";
        }
    }
}
