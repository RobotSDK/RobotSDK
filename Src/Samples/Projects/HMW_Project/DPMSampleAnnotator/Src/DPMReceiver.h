#ifndef DPMRECEIVER
#define DPMRECEIVER

//=================================================
//Please add headers here:
#include<libdpm_ocv/ImageObjects.h>
#include<rosinterface.h>
#include<opencv2/opencv.hpp>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS DPMReceiver

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
public:
    ADD_PARAM(QString, category, "car")
};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(QString, rostopic, "car_pixel_xy")
    ADD_VAR(u_int32_t, rosqueuesize, 1000)
    ADD_VAR(int, rosqueryinterval, 10)
    ADD_VAR(bool, idflag, 0)
    ADD_VAR(int, curid, 0)
public:
    typedef ROSSub<libdpm_ocv::ImageObjectsConstPtr> subtype;
    ADD_INTERNAL_QOBJECT_TRIGGER(subtype, dpmsub, 1, rostopic, rosqueuesize, rosqueryinterval)
    ADD_INTERNAL_DEFAULT_CONNECTION(dpmsub, receiveMessageSignal)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    double rostimestamp;
    QVector<cv::Rect> detection;
    QVector<int> id;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
