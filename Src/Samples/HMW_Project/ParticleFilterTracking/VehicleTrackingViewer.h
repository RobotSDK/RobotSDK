#ifndef VEHICLETRACKINGVIEWER
#define VEHICLETRACKINGVIEWER

//=================================================
//Please add headers here:
#include<VehicleTracker.h>
#include<VirtualScanGlobalizer.h>
#include<VehicleTrackingViewerWidget.h>
#include<sync.h>
#include<QListWidget>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS VehicleTrackingViewer

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 2

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 0

//=================================================
//Params types configuration

//If you need to refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(uint, maxrange, 80)
    ADD_PARAM(uint, gridsize, 10)
    ADD_PARAM(uint, imagesize, 600)
};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_QWIDGET(VehicleTrackingViewerWidget, viewer)
    ADD_QLAYOUT(QHBoxLayout, layout)
    ADD_QWIDGET(QListWidget, list)
    ADD_CONNECTION(list,currentRowChanged,viewer,slotShowRect,int)
public:
    ADD_SYNC(sync,1)
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
