#ifndef VELODYNESENSOR
#define VELODYNESENSOR

//=================================================
//Please add headers here:
#include<pcl/point_cloud.h>
#include<pcl/point_types.h>
#include<opencv2/opencv.hpp>
#include<sensor_msgs/PointCloud2.h>
#include<rosinterface.h>

//=================================================
#include<RobotSDK.h>
//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS VelodyneSensor

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
    ADD_PARAM(QString, calibfilename, "#(VelodyneCalibFileName)")
};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    cv::Mat extrinsicmat;
public:
    ADD_VAR(QString, topic, "/velodyne_points")
    ADD_VAR(u_int32_t, queuesize, 1000)
    ADD_VAR(int, queryinterval, 10)
public:
    typedef ROSSub<sensor_msgs::PointCloud2ConstPtr> rossub;
    ADD_INTERNAL_QOBJECT_TRIGGER(rossub, velodynesub, 1, topic,queuesize,queryinterval)
    ADD_INTERNAL_DEFAULT_CONNECTION(velodynesub, receiveMessageSignal)
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    cv::Mat extrinsicmat;
public:
    sensor_msgs::PointCloud2ConstPtr rospoints;
    pcl::PointCloud<pcl::PointXYZI>::Ptr pclpoints=pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
};

//=================================================
//You can declare functions here


//=================================================

#endif
