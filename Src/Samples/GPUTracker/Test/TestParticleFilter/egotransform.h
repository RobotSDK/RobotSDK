#ifndef EGOTRANSFORM_H
#define EGOTRANSFORM_H

#include<cuda.h>
#include<cuda_runtime.h>

#include"def.h"

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
    EgoTransform(StateValueType x1, StateValueType y1, StateValueType theta1, StateValueType x2, StateValueType y2, StateValueType theta2);
    __host__ __device__
    EgoTransform(StateValueType * transformMatrix);
public:
    __host__ __device__
    void transformState2D(StateValueType & x, StateValueType & y, StateValueType & theta);
    __host__ __device__
    void transformState3D(StateValueType * stateMatrix);
};

#endif // EGOTRANSFORM_H
