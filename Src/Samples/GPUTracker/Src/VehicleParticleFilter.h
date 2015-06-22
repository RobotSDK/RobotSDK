#ifndef VEHICLEPARTICLEFILTER_H
#define VEHICLEPARTICLEFILTER_H

#include"particlefilterdef.h"

STATE_DEF(Vehicle, 6, x,y,theta,v,width,length)

struct ObstacleMap
{
    int mapsize;
    float gridsize;
    unsigned char * map;
    float wtable[3]={2.0f,1.0f,0.4f};
    int edgepointnum=5;
    float margin=0.1;
};

MEASUREDATA_DEF(Vehicle,ObstacleMap)

PARTICLE_FILTER_INTERACT_FUNCS_DECL(Vehicle)

#endif // VEHICLEPARTICLEFILTER_H

