#include "egotransform.h"

__host__ __device__
EgoTransform::EgoTransform()
{
    transform.deltax=0;
    transform.deltay=0;
    transform.deltatheta=0;
}

__host__ __device__
EgoTransform::EgoTransform(StateValueType dx, StateValueType dy, StateValueType theta1, float theta2)
{
    StateValueType deltax=-dx;
    StateValueType deltay=-dy;
    transform.deltax=deltax*cos(theta2)+deltay*sin(theta2);
    transform.deltay=-deltax*sin(theta2)+deltay*cos(theta2);
    transform.deltatheta=theta1-theta2;
}

__host__ __device__
EgoTransform::EgoTransform(StateValueType * transformMatrix)
{
    int i,num=sizeof(transform.matrix)/sizeof(StateValueType);
    for(i=0;i<num;i++)
    {
        transform.matrix[i]=transformMatrix[i];
    }
}

__host__ __device__
void EgoTransform::transformState2D(StateValueType & x, StateValueType & y, StateValueType & theta)
{

    StateValueType tmpx=x*cos(transform.deltatheta)-y*sin(transform.deltatheta)+transform.deltax;
    StateValueType tmpy=x*sin(transform.deltatheta)+y*cos(transform.deltatheta)+transform.deltay;
    x=tmpx;
    y=tmpy;
    theta+=transform.deltatheta;
}

__host__ __device__
void EgoTransform::transformState3D(StateValueType * stateMatrix)
{
    StateValueType tmpmatrix[16];
    tmpmatrix[0]=transform.matrix[0]*stateMatrix[0]+transform.matrix[1]*stateMatrix[4]+transform.matrix[2]*stateMatrix[8]+transform.matrix[3]*stateMatrix[12];
    tmpmatrix[1]=transform.matrix[0]*stateMatrix[1]+transform.matrix[1]*stateMatrix[5]+transform.matrix[2]*stateMatrix[9]+transform.matrix[3]*stateMatrix[13];
    tmpmatrix[2]=transform.matrix[0]*stateMatrix[2]+transform.matrix[1]*stateMatrix[6]+transform.matrix[2]*stateMatrix[10]+transform.matrix[3]*stateMatrix[14];
    tmpmatrix[3]=transform.matrix[0]*stateMatrix[3]+transform.matrix[1]*stateMatrix[7]+transform.matrix[2]*stateMatrix[11]+transform.matrix[3]*stateMatrix[15];

    tmpmatrix[4]=transform.matrix[4]*stateMatrix[0]+transform.matrix[5]*stateMatrix[4]+transform.matrix[6]*stateMatrix[8]+transform.matrix[7]*stateMatrix[12];
    tmpmatrix[5]=transform.matrix[4]*stateMatrix[1]+transform.matrix[5]*stateMatrix[5]+transform.matrix[6]*stateMatrix[9]+transform.matrix[7]*stateMatrix[13];
    tmpmatrix[6]=transform.matrix[4]*stateMatrix[2]+transform.matrix[5]*stateMatrix[6]+transform.matrix[6]*stateMatrix[10]+transform.matrix[7]*stateMatrix[14];
    tmpmatrix[7]=transform.matrix[4]*stateMatrix[3]+transform.matrix[5]*stateMatrix[7]+transform.matrix[6]*stateMatrix[11]+transform.matrix[7]*stateMatrix[15];

    tmpmatrix[8]=transform.matrix[8]*stateMatrix[0]+transform.matrix[9]*stateMatrix[4]+transform.matrix[10]*stateMatrix[8]+transform.matrix[11]*stateMatrix[12];
    tmpmatrix[9]=transform.matrix[8]*stateMatrix[1]+transform.matrix[9]*stateMatrix[5]+transform.matrix[10]*stateMatrix[9]+transform.matrix[11]*stateMatrix[13];
    tmpmatrix[10]=transform.matrix[8]*stateMatrix[2]+transform.matrix[9]*stateMatrix[6]+transform.matrix[10]*stateMatrix[10]+transform.matrix[11]*stateMatrix[14];
    tmpmatrix[11]=transform.matrix[8]*stateMatrix[3]+transform.matrix[9]*stateMatrix[7]+transform.matrix[10]*stateMatrix[11]+transform.matrix[11]*stateMatrix[15];

    tmpmatrix[12]=0;
    tmpmatrix[13]=0;
    tmpmatrix[14]=0;
    tmpmatrix[15]=1;

    for(int i=0;i<16;i++)
    {
        stateMatrix[i]=tmpmatrix[i];
    }
}
