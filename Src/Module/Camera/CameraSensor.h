#ifndef CAMERASENSOR
#define CAMERASENSOR

//=================================================
//Please add headers here:
#include<sensor_msgs/Image.h>
#include<opencv2/opencv.hpp>
#include<rosinterface.h>
#include<QFile>

//=================================================
#include<RobotSDK.h>
//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS CameraSensor

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
public:
    ADD_PARAM(QString, calibfilename, "#(CameraCalibFileName)")
};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    cv::Mat extrinsicmat;
    cv::Mat cameramat;
    cv::Mat distcoeff;
public:
    ADD_VAR(QString, topic, "/image_raw")
    ADD_VAR(u_int32_t, queuesize, 1000)
    ADD_VAR(int, queryinterval, 10)
public:
    typedef ROSSub<sensor_msgs::ImageConstPtr> rossub;
    ADD_INTERNAL_QOBJECT_TRIGGER(rossub, camerasub, 1, topic,queuesize,queryinterval)
    ADD_INTERNAL_DEFAULT_CONNECTION(camerasub, receiveMessageSignal)
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    sensor_msgs::ImageConstPtr rosimage;
    cv::Mat cvimage;
public:
    cv::Mat extrinsicmat;
    cv::Mat cameramat;
    cv::Mat distcoeff;
};

//=================================================
//You can declare functions here


//=================================================

#endif
