#ifndef ROSBAGLOADER
#define ROSBAGLOADER

//=================================================
//Please add headers here:
#include<ros/ros.h>
#include<rosbag/bag.h>
#include<rosbag/view.h>
#include<rosbag/query.h>
#include<sensor_msgs/Image.h>
#include<sensor_msgs/CameraInfo.h>
#include<opencv2/opencv.hpp>
#include<cv_bridge/cv_bridge.h>
#include<rosinterface.h>
#include"DPMModifier.h"

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS ROSBagLoader

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need to refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(QString, bagfilename, "")
    ADD_PARAM(QString, bagimagetopic, "/image_raw")
    ADD_PARAM(bool, imageprocflag, 0)
    ADD_PARAM(QString, bagcaminfotopic, "/camera_info")
    ADD_PARAM(uint, bagstart, 0)
    ADD_PARAM(uint, baginterval, 1)
    ADD_PARAM(bool, rgbinvertflag, 0)
    ADD_PARAM(QString, ROI, "0,0,0,0")
};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    rosbag::Bag bag;
    rosbag::View * view=NULL;
    rosbag::View::const_iterator viewiter;
    int curframe;
    int x,y,width,height;
public:
    ADD_VAR(QString, rosimagetopic, "/image_raw")
    ADD_VAR(QString, roscaminfotopic, "/camera_info")
    ADD_VAR(QString, rosreceiveimagetopic, "/image_raw")
    ADD_VAR(u_int32_t, rosqueuesize, 1000)
    ADD_VAR(int, rosqueryinterval, 10)
public:
    typedef ROSPub<sensor_msgs::Image> pubtype;
    ADD_INTERNAL_QOBJECT_TRIGGER(pubtype, imagepub, 0, rosimagetopic, rosqueuesize)
    typedef ROSPub<sensor_msgs::CameraInfo> caminfopubtype;
    ADD_INTERNAL_QOBJECT_TRIGGER(caminfopubtype, caminfopub, 0, roscaminfotopic, rosqueuesize)
    typedef ROSSub<sensor_msgs::ImageConstPtr> subtype;
    ADD_INTERNAL_QOBJECT_TRIGGER(subtype, imagesub, 0, rosreceiveimagetopic, rosqueuesize, rosqueryinterval)
    ADD_INTERNAL_DEFAULT_CONNECTION(imagesub,receiveMessageSignal)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    int frameid;
    cv::Mat image;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
