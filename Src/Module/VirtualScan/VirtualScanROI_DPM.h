#ifndef VIRTUALSCANROI_DPM
#define VIRTUALSCANROI_DPM

//=================================================
//Please add headers here:
#include<VirtualScanGenerator.h>
#include<DPMDetector.h>
#include<CameraSensor.h>
#include<QVector>
#include<QPair>
#include<sync.h>
#include<QPoint>
#include<QRect>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS VirtualScanROI_DPM

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 3

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
    ADD_VAR(double, minrange, 2)
    ADD_VAR(double, maxrange, 60)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    QVector< QPair<int, int> > roi;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
