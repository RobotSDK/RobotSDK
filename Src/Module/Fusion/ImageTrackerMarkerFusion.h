#ifndef IMAGETRACKERMARKERFUSION
#define IMAGETRACKERMARKERFUSION

//=================================================
//Please add headers here:
#include<CameraSensor.h>
#include<TrackerMarkerReceiver.h>
#include<sync.h>
#include<QPointF>
#include<geometry_msgs/Point.h>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS ImageTrackerMarkerFusion

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 2

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
    ADD_SYNC(sync,1)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    QTime trackertimestamp;
    cv::Mat cvimage;
    QVector<QPointF> corners;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
