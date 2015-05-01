#ifndef VIRTUALSCANCLUSTER
#define VIRTUALSCANCLUSTER

//=================================================
//Please add headers here:
#include<VirtualScanGenerator.h>
#include<QQueue>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS VirtualScanCluster

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : INHERIT_NODE_PARAMS_TYPE(VirtualScanGenerator)
{
public:
    ADD_PARAM(uint, neighbornum, 5)
    ADD_PARAM(uint, minpointsnum, 10)
    ADD_PARAM(double, xsigma, 0.005)
    ADD_PARAM(double, xminsigma, 0.1)
    ADD_PARAM(double, ysigma, 0.05)
    ADD_PARAM(double, yminsigma, 0.1)
    ADD_PARAM(double, threshold, 0.08)
};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public RobotSDK::NODE_VARS_BASE_TYPE
{

};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
NODE_DATA_TYPE_REF(VirtualScanGenerator)

//=================================================
//You can declare functions here


//=================================================
}

#endif
