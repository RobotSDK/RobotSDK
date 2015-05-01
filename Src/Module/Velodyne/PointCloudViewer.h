#ifndef POINTCLOUDVIEWER
#define POINTCLOUDVIEWER

//=================================================
//Please add headers here:
#include<glviewer.h>
#include"VelodyneSensor.h"

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS PointCloudViewer

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 0

//=================================================
//Params types configuration

//If you need refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public RobotSDK::NODE_PARAMS_BASE_TYPE
{

};

//=================================================
//Vars types configuration

//If you need refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public RobotSDK::NODE_VARS_BASE_TYPE
{
public:
    GLuint velodynelist;
public:
    ADD_QLAYOUT(QHBoxLayout, layout)
    ADD_QWIDGET(QTabWidget, tabwidget)
    ADD_QWIDGET(RobotSDK::GLViewer, viewer)
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public RobotSDK::NODE_DATA_BASE_TYPE
{

};

//=================================================
//You can declare functions here


//=================================================
}

#endif
