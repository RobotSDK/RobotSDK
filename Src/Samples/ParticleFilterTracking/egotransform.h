#ifndef EGOTRANSFORM_H
#define EGOTRANSFORM_H

#include<cuda.h>
#include<cuda_runtime.h>

#include"particlefilterdef.h"

class EgoTransform  //M_t+1^-1 * M_t
{
public:
    union
    {
        struct
        {
            StateValueType deltax;
            StateValueType deltay;
            StateValueType deltatheta;
        };
        StateValueType matrix[16];
    }transform;
public:
    __host__ __device__
    EgoTransform();
    __host__ __device__
    EgoTransform(StateValueType dx, StateValueType dy, StateValueType theta1, StateValueType theta2);
    __host__ __device__
    EgoTransform(StateValueType * transformMatrix);
public:
    __host__ __device__
    void transformState2D(StateValueType & x, StateValueType & y, StateValueType & theta);
    __host__ __device__
    void transformState3D(StateValueType * stateMatrix);
};

#endif // EGOTRANSFORM_H
