#ifndef CONTROLLER
#define CONTROLLER

//=================================================
//Please add headers here:
#include<Localization.h>
#include<Planning.h>
#include<Obstacle.h>

#include<SHSpur.h>
#include<math.h>
#include<QTimer>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS Controller

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 3

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need to refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(double, linear_velocity, 1.0)
    ADD_PARAM(double, angular_velocity, 1.0)
};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(double, pos_error, 0.2)
    ADD_VAR(double, ang_error, 0.04)
    bool checkposflag;
    bool checkangflag;
    double destposx, destposy;
    double destang;
    bool obstacleflag;
};

//=================================================
//Data types configuration
enum OrderState
{
    None,
    Process,
    Finish,
    Wait
};
//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
     OrderState state;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
