#ifndef VIRTUALSCANGENERATOR
#define VIRTUALSCANGENERATOR

//=================================================
//Please add headers here:
#include<VelodyneSensor.h>
#include<opencv2/opencv.hpp>
#include<sensor_msgs/LaserScan.h>
#include<fastvirtualscan.h>
#include<QMultiMap>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS VirtualScanGenerator

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(uint, beamnum, 1000)
    ADD_PARAM(double, heightstep, 0.3)
    ADD_PARAM(double, slope, 30)
    ADD_PARAM(double, minfloor, -3)
    ADD_PARAM(double, maxceiling, 3)
    ADD_PARAM(double, maxfloor, -1.2)
    ADD_PARAM(double, minceiling, -0.5)
    ADD_PARAM(double, passheight, 2)
    ADD_PARAM(double, rotation, 3)
    ADD_PARAM(double, minrange, 0.5)
};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    FastVirtualScan virtualscan;
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
public:
    QVector<double> virtualscan;
    QVector<double> minheights;
    QVector<double> maxheights;
    QVector<uint> labels;
    uint clusternum;
    QMultiMap<uint, uint> clusters;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
