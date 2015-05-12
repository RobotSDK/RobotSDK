#ifndef IMAGEPROCESSOR
#define IMAGEPROCESSOR

//=================================================
//Please add headers here:
#include"CameraSensor.h"

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS ImageProcessor

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need refer params type of other node class, please uncomment below and comment its own params type.
NODE_PARAMS_TYPE_REF(CameraSensor)

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR_WITH_OPTIONS(double, rotation, 0, QList<double>()<<0<<90<<180<<270)
    ADD_VAR(double, scale, 1.0)
    ADD_VAR(double, alpha, 1)
    ADD_VAR(double, beta, 0)
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
NODE_DATA_TYPE_REF(CameraSensor)

//=================================================
//You can declare functions here

//=================================================
}

#endif
