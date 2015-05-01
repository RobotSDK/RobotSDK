#ifndef CAMERAVIRTUALSCANFUSION
#define CAMERAVIRTUALSCANFUSION

//=================================================
//Please add headers here:
#include<CameraSensor.h>
#include<VirtualScanGenerator.h>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS CameraVirtualScanFusion

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 2

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public RobotSDK::NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(double, minrange,0)
    ADD_PARAM(double, maxrange, 100)
};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public RobotSDK::NODE_VARS_BASE_TYPE
{
public:
    bool initialcamera;
    cv::Mat cameraextrinsicmat;
    cv::Mat cameramat;
    cv::Mat distcoeff;
    bool initialvirtualscan;
    cv::Mat virtualscanextrinsicmat;
public:
    ADD_SYNC(sync, 0)
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public RobotSDK::NODE_DATA_BASE_TYPE
{
public:
    QTime virtualscantimestamp;
    double minrange;
    double maxrange;
    cv::Mat cvimage;
    QVector< double > virtualscan;
    QVector< double > minheights;
    QVector< double > maxheights;
    QVector< uint > labels;
    uint clusternum;
    QMultiMap<uint, uint> clusters;
    QMultiMap< uint, QPair< QPoint, QPoint > > stixel;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
