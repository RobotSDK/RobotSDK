#ifndef PARTICLEFILTER_H
#define PARTICLEFILTER_H

#include"particlefilterdef.h"

STATE_DEF(Vehicle, 6, x,y,theta,v,width,length)

struct TargetVehicle
{
    STATE_TYPE(Vehicle) state;
};

MEASUREDATA_DEF(Vehicle,TargetVehicle)

PARTICLE_FILTER_INTERACT_FUNCS_DECL(Vehicle)

#endif // PARTICLEFILTERTYPE_H

