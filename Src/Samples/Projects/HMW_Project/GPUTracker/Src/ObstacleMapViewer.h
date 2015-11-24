#ifndef OBSTACLEMAPVIEWER
#define OBSTACLEMAPVIEWER

//=================================================
//Please add headers here:
#include"ObstacleMapGenerator.h"
#include<QTabWidget>
#include<QScrollArea>
#include<QLabel>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS ObstacleMapViewer

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 0

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
    ADD_QLAYOUT(QHBoxLayout, layout)
    ADD_QWIDGET(QTabWidget, tabwidget)
    ADD_QWIDGET(QScrollArea, scrollarea)
    ADD_QWIDGET(QLabel, viewer, "Obstacle Map Viewer")
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{

};

//=================================================
//You can declare functions here


//=================================================
}

#endif
