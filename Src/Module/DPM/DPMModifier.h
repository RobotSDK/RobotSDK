#ifndef DPMMODIFIER
#define DPMMODIFIER

//=================================================
//Please add headers here:
#include"CameraSensor.h"
#include"DPMModifier.h"
#include"DPMModifierWidgets.h"
//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS DPMModifier

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 2

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

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
    cv::Mat image;
    QGraphicsPixmapItem * pixmap;
public:
    ADD_QLAYOUT(QVBoxLayout, layout)
    ADD_QWIDGET(QTabWidget, tabwidget)
    ADD_QWIDGET(DPMViewer, viewer)
public:
    ADD_INTERNAL_QWIDGET_TRIGGER(QPushButton, apply, "Output Change")
    ADD_INTERNAL_DEFAULT_CONNECTION(apply, clicked)
    ADD_QLAYOUT(QHBoxLayout, buttonlayout)
public:
    ADD_SYNC(dpmsync, 1)
};

//=================================================
//Data types configuration

//If you need refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    cv::Mat cvimage;
    QVector<cv::Rect> detection;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
