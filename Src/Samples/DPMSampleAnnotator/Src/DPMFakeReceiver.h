#ifndef DPMFAKERECEIVER
#define DPMFAKERECEIVER

//=================================================
//Please add headers here:
#include<rosinterface.h>
#include<sensor_msgs/Image.h>
#include"DPMReceiver.h"

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS DPMFakeReceiver

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 0

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need to refer params type of other node class, please uncomment below and comment its own params type.
NODE_PARAMS_TYPE_REF(DPMReceiver)

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(QString, rostopic, "/image_raw")
    ADD_VAR(u_int32_t, rosqueuesize, 1000)
    ADD_VAR(int, rosqueryinterval, 10)
public:
    typedef ROSSub<sensor_msgs::ImageConstPtr> subtype;
    ADD_INTERNAL_QOBJECT_TRIGGER(subtype, dpmsub, 1, rostopic, rosqueuesize, rosqueryinterval)
    ADD_INTERNAL_DEFAULT_CONNECTION(dpmsub, receiveMessageSignal)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
NODE_DATA_TYPE_REF(DPMReceiver)

//=================================================
//You can declare functions here


//=================================================
}

#endif
