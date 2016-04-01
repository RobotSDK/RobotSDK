#ifndef DPMSAMPLELOADER
#define DPMSAMPLELOADER

//=================================================
//Please add headers here:
#include"DPMAnnotator.h"
#include<QFile>
#include<QFileInfo>
#include<QTextStream>
#include<opencv2/opencv.hpp>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS DPMSampleLoader

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
    ADD_PARAM(QString, rosbagfile, "")
    ADD_PARAM(QString, samplefilebasename, "")
};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    QFile file;
    QTextStream stream;
    QString imagesdir;
public:
    ADD_VAR(int, imagefilenamewidth, 5)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    cv::Mat image;
    double rostimestamp;
    int frameid;
    QString category;
    int id;
    cv::Rect rect;
    QString attributes;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
