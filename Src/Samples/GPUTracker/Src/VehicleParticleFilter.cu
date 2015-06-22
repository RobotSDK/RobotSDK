#include"VehicleParticleFilter.cuh"

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

#define UPDATEWEIGHT(ctheta,stheta,statex,statey,mapsize,gridsize,obmap,x,y,posid,wtable,weight) \
    if(weight>0) \
    { \
        float gx=ctheta*x-stheta*y+statex; \
        float gy=stheta*x+ctheta*y+statey; \
        int mx=mapsize/2-int(gy/gridsize+0.5f); \
        int my=mapsize/2-int(gx/gridsize+0.5f); \
        if(mx>=0&&mx<mapsize&&my>=0&&my<mapsize) \
        { \
            int mapid=(my*mapsize+mx)*3; \
            if(obmap[mapid+2]==255) \
            { \
                weight*=wtable[1]; \
            } \
            else if(obmap[mapid+posid]==255) \
            { \
                weight*=wtable[0]; \
            } \
            else \
            { \
                weight*=wtable[2]; \
            } \
        } \
        else \
        { \
            weight=0; \
        } \
    }

PARTICLE_MEASURE_FUNC(Vehicle,state,measureData)
{
    float widthstep=state.width/(measureData.edgepointnum+1);
    float lengthstep=state.length/(measureData.edgepointnum+1);
    float c=cos(state.theta);
    float s=sin(state.theta);
    float cornerx=state.width/2;
    float cornery=state.length/2;
    float weight=1;

    for(int i=1;i<=measureData.edgepointnum&&weight>0;i++)
    {
        float w=-cornerx+i*widthstep;

        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,w,cornery,0,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,w,cornery+measureData.margin,1,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,w,cornery-measureData.margin,2,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,w,-cornery,0,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,w,-cornery-measureData.margin,1,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,w,-cornery+measureData.margin,2,measureData.wtable,weight);

        float l=-cornery+i*lengthstep;

        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,cornerx,l,0,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,cornerx+measureData.margin,l,1,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,cornerx-measureData.margin,l,2,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,-cornerx,l,0,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,-cornerx-measureData.margin,l,1,measureData.wtable,weight);
        UPDATEWEIGHT(c,s,state.x,state.y,measureData.mapsize,measureData.gridsize,measureData.map,-cornerx+measureData.margin,l,2,measureData.wtable,weight);
    }
    return weight;
}

PARTICLE_FILTER_MEASURE_FUNC(Vehicle, measureData)
{
    MEASUREDATA_TYPE(Vehicle) obstaclemap=measureData;
    size_t mapdatasize=obstaclemap.mapsize*obstaclemap.mapsize*3*sizeof(unsigned char);
    cudaMalloc((void **)(&(obstaclemap.map)),mapdatasize);
    cudaMemcpy(obstaclemap.map,measureData.map,mapdatasize,cudaMemcpyHostToDevice);
    Vehicle_Base::measureParticles(obstaclemap);
    cudaDeviceSynchronize();
    cudaFree(obstaclemap.map);
}

PARTICLE_FILTER_INTERACT_FUNCS(Vehicle)
