#ifndef LINEEXTRACTOR
#define LINEEXTRACTOR

//=================================================
//Please add headers here:
#include<mrpt/math/ransac_applications.h>
#include<VirtualScanGenerator.h>
#include<VirtualScanROI_DPM.h>
#include<QVector>
#include<QPointF>
#include<QLineF>
#include<sync.h>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS LineExtractor

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
public:
    ADD_PARAM(double, dist_threshold, 0.2)
    ADD_PARAM(int, min_inliers, 10)
};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_SYNC(sync,0)
public:
    ADD_VAR(double, cluster_threshold, )
};

//=================================================
//Data types configuration

struct ExtractedLines
{
    int id;
    double starttheta;
    double endtheta;
    QVector<QLineF> lines;
    QVector<QPointF> points;
};

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    QVector<ExtractedLines> lines;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
