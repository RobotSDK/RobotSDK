#ifndef IMAGEPROCESSOR
#define IMAGEPROCESSOR

//=================================================
//Please add headers here:
#include"CameraSensor.h"
//=================================================

#include<RobotSDK.h>

//=================================================
//Port configuration

#undef NODE_CLASS
#define NODE_CLASS ImageProcessor

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraSensor)

//=================================================
//Params types configuration

//If you need refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM_WITH_OPTIONS(double, angle, 0, QList<double>()<<0<<90<<180<<270)
    ADD_PARAM(double, ratio, 1.0)
    ADD_PARAM(double, alpha, 1)
    ADD_PARAM(double, beta, 0)
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
NODE_DATA_TYPE_REF(CameraSensor)

//=================================================
//You can declare functions here

//=================================================

#endif
