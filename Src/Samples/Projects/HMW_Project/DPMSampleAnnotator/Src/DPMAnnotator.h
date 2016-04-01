#ifndef DPMANNOTATOR
#define DPMANNOTATOR

//=================================================
//Please add headers here:
#include"DPMSampleLoader.h"
#include"DPMModifier.h"
#include"DPMAnnotatorWidget.h"
#include<QFile>
#include<QTextStream>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS DPMAnnotator

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 2

//=================================================
//Params types configuration

//If you need to refer params type of other node class, please uncomment below and comment its own params type.
NODE_PARAMS_TYPE_REF(DPMModifier)

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    QStringList categories;
public:
    bool dataflag;
    QString rosbagfile;
    double rostimestamp;
    int frameid;
    QString category;
    int id;
    cv::Rect rect;
public:
    ADD_VAR(double, alpha, 1)
    ADD_VAR(double, beta, 0)
public:
    ADD_VAR(QString, attributesfile, "")
public:
    ADD_INTERNAL_QWIDGET_TRIGGER(DPMAnnotatorWidget, annotator)
    ADD_INTERNAL_DEFAULT_CONNECTION(annotator, signalNext)
    ADD_QLAYOUT(QHBoxLayout, layout)
public:
    ADD_VAR(bool, rgbconvertflag, 0)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
NODE_DATA_TYPE_REF(DPMModifier)

//=================================================
//You can declare functions here


//=================================================
}

#endif
