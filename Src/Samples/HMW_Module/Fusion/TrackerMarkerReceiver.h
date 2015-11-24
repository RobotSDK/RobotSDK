#ifndef TRACKERMARKERRECEIVER
#define TRACKERMARKERRECEIVER

//=================================================
//Please add headers here:
#include<visualization_msgs/MarkerArray.h>
#include<rosinterface.h>
#include<opencv2/opencv.hpp>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS TrackerMarkerReceiver

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
    ADD_VAR(QString, calibfilename, "#(VelodyneCalibFileName)")
public:
    cv::Mat extrinsicmat;
public:
    ADD_VAR(QString, topic, "/trackercubic")
    ADD_VAR(u_int32_t, queuesize, 1000)
    ADD_VAR(int, queryinterval, 10)
public:
    typedef ROSSub<visualization_msgs::MarkerConstPtr> rossub;
    ADD_INTERNAL_QOBJECT_TRIGGER(rossub, trackersub, 1, topic,queuesize,queryinterval)
    ADD_INTERNAL_DEFAULT_CONNECTION(trackersub, receiveMessageSignal)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    cv::Mat extrinsicmat;
public:
    visualization_msgs::MarkerConstPtr tracker;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
