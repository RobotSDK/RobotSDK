#ifndef VIRTUALSCANPUBLISHER
#define VIRTUALSCANPUBLISHER

//=================================================
//Please add headers here:
#include<VirtualScanGenerator.h>
#include<sensor_msgs/LaserScan.h>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS VirtualScanPublisher

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 0

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
    ADD_VAR(QString, topicpointcloud2, "/virtualscan")
    ADD_VAR(QString, topiclaserscan, "/laserscan")
    ADD_VAR(u_int32_t, queuesize, 1000)
    ADD_VAR(bool, pub3dflag, 1)
public:
    typedef ROSPub<sensor_msgs::PointCloud2> rospubpointcloud2;
    ADD_INTERNAL_QOBJECT_TRIGGER(rospubpointcloud2, virtualscanpubpointcloud2, 0, topicpointcloud2, queuesize)
    typedef ROSPub<sensor_msgs::LaserScan> rospublaserscan;
    ADD_INTERNAL_QOBJECT_TRIGGER(rospublaserscan, virtualscanpublaserscan, 0, topiclaserscan, queuesize)
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{

};

//=================================================
//You can declare functions here


//=================================================
}

#endif
