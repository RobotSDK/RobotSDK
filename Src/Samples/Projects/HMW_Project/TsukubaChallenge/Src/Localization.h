#ifndef LOCALIZATION
#define LOCALIZATION

//=================================================
//Please add headers here:
#include<rosinterface.h>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS Localization

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 0

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need to refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{

};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(QString, destinationframe, "world")
    ADD_VAR(QString, originalframe, "odm_frame")
    ADD_VAR(int, queryinterval, 10)
public:
    ADD_INTERNAL_QOBJECT_TRIGGER(ROSTFSub, odmsub, 1, destinationframe, originalframe, queryinterval)
    ADD_INTERNAL_DEFAULT_CONNECTION(odmsub, receiveTFSignal)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    double posx;
    double posy;
    double theta;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
