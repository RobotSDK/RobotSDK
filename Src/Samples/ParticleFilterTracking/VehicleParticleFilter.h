#ifndef VEHICLEPARTICLEFILTER_H
#define VEHICLEPARTICLEFILTER_H

#include"particlefilterdef.h"

STATE_DEF(Vehicle, 8, x, y, theta, v, width, length, wsigma, lsigma)

struct VirtualScanData
{
    int beamnum;
    double * beams;
};

MEASUREDATA_DEF(Vehicle,VirtualScanData)

PARTICLE_FILTER_INTERACT_FUNCS_DECL(Vehicle)

#endif // VEHICLEPARTICLEFILTER_H

