#ifndef VEHICLEPARTICLEFILTER_H
#define VEHICLEPARTICLEFILTER_H

#include"particlefilterdef.h"

STATE_DEF(Vehicle, 9, x,y,theta,thetaoffset,v,width,length,wsigma,lsigma)

struct ObstacleMap
{
    int mapsize;
    float gridsize;
    float radius;
    unsigned char * map;
    float * mapdata;
    float wtable[3]={20.0f,1.0f,0.1f};
    float sigma;
};

MEASUREDATA_DEF(Vehicle,ObstacleMap)

PARTICLE_FILTER_INTERACT_FUNCS_DECL(Vehicle)

#endif // VEHICLEPARTICLEFILTER_H

