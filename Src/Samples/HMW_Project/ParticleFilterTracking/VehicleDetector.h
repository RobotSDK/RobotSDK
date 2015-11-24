#ifndef VEHICLEDETECTOR
#define VEHICLEDETECTOR

//=================================================
//Please add headers here:
#include<VirtualScanGlobalizer.h>
#include<VehicleParticleFilter.h>
#include<VehicleDetectorWidget.h>
#include<QImage>
#include<QPainter>
#include<QPixmap>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS VehicleDetector

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
    ADD_QWIDGET(VehicleDetectorWidget,detector)
    ADD_QLAYOUT(QHBoxLayout,layout)
public:
    QTime timestamp;
    int idcount;
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    std::vector<int> objectids;
    std::vector<STATE_TYPE(Vehicle)> objectstates;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
