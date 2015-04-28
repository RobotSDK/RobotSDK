#ifndef VIRTUALSCANCLUSTER
#define VIRTUALSCANCLUSTER

//=================================================
//Please add headers here:
#include<VirtualScanGenerator.h>

//=================================================
#include<RobotSDK.h>
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
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(double, neighbordis, 0.3)
    ADD_PARAM(uint, minpointsnum, 10)
};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{

};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
NODE_DATA_TYPE_REF(VirtualScanGenerator)

//=================================================
//You can declare functions here


//=================================================

#endif
