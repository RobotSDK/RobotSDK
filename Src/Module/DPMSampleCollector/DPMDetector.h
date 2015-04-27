#ifndef DPMDETECTOR
#define DPMDETECTOR

//=================================================
//Please add headers here:
#include<dpm/ImageObjects.h>
#include<rosinterface.h>
#include<opencv2/opencv.hpp>
//=================================================
#include<RobotSDK.h>
//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS DPMDetector

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 0

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{

};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(QString, topic, "car_pos_xy")
    ADD_VAR(u_int32_t, queuesize, 1000)
    ADD_VAR(int, queryinterval, 10)
public:
    typedef ROSSub<dpm::ImageObjectsConstPtr> rossub;
    ADD_INTERNAL_QOBJECT_TRIGGER(rossub, dpmsub, 1, topic, queuesize, queryinterval)
    ADD_INTERNAL_DEFAULT_CONNECTION(dpmsub, receiveMessageSignal)
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    dpm::ImageObjectsConstPtr rosdetection;
    QVector<cv::Rect> detection;
};

//=================================================
//You can declare functions here


//=================================================

#endif
