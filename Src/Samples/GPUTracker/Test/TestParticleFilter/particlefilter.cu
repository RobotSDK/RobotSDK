#include"particlefilter.cuh"

PARTICLE_INITIALIZE_FUNC(Vehicle,initialState,randomOffset)
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

PARTICLE_RANDOMNIZE_FUNC(Vehicle,randomOffset)
{
    state.v+=randomOffset.v;
    state.theta+=randomOffset.theta;
}

PARTICLE_UPDATE_FUNC(Vehicle,deltaMsec)
{
    float msec=deltaMsec/1000.0;
    float dis=msec*state.v;
    state.x+=dis*cos(state.theta);
    state.y+=dis*sin(state.theta);
}

PARTICLE_TRANSFORM_FUNC(Vehicle,transform)
{
    transform.transformState2D(state.x,state.y,state.theta);
}

PARTICLE_MEASURE_FUNC(Vehicle,state,measureData)
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

PARTICLE_FILTER_MEASURE_DEFAULT_FUNC(Vehicle)

PARTICLE_FILTER_INTERACT_FUNCS(Vehicle)
