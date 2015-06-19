#ifndef DPMMODIFIER_H
#define DPMMODIFIER_H

//=================================================
//Please add headers here:
#include"DPMReceiver.h"
#include"ROSBagLoader.h"
#include"DPMModifierWidget.h"
#include<QLayout>
#include<QPushButton>
#include<QTabWidget>
#include<QGraphicsPixmapItem>

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
#define OUTPUT_PORT_NUM 2

//=================================================
//Params types configuration

#define DIRECTIONSTRING "front,front-left,front-right,left,right,back,back-left,back-right"

//If you need to refer params type of other node class, please uncomment below and comment its own params type.
//NODE_PARAMS_TYPE_REF(RefNodeClassName)
class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(QString, categories, "car,pedestrian")
};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    QGraphicsPixmapItem * pixmap;
    QStringList categories;
public:
    bool dataflag;
    QTime timestamp;
    QString rosbagfile;
    int frameid;
    cv::Mat image;
    bool rgbinvertflag;
public:
    ADD_QLAYOUT(QHBoxLayout, layout)
    ADD_QLAYOUT(QVBoxLayout, controllayout)
    ADD_QWIDGET(QTabWidget, tabwidget)
    ADD_INTERNAL_QWIDGET_TRIGGER(DPMModifierWidget, dpmmodifier)
    ADD_QWIDGET(DPMController, dpmcontroller)
    ADD_CONNECTION(dpmcontroller,signalSetFilter,dpmmodifier,slotSetFilter,QMap<QString, bool>)
    ADD_INTERNAL_DEFAULT_CONNECTION(dpmmodifier,signalNext)
    ADD_INTERNAL_QWIDGET_TRIGGER(QPushButton, trigger, "Next")
    ADD_INTERNAL_DEFAULT_CONNECTION(trigger,clicked)
    ADD_VAR(int, automsec,1000)
    ADD_QWIDGET(QPushButton, autostart, "Auto Start")
    ADD_QWIDGET(QPushButton, autostop, "Auto Stop")
    ADD_INTERNAL_QOBJECT_TRIGGER(QTimer, triggertimer, 0)
    ADD_INTERNAL_DEFAULT_CONNECTION(triggertimer,timeout)
    ADD_CONNECTION(autostart,clicked,triggertimer,start)
    ADD_CONNECTION(autostop,clicked,triggertimer,stop)
public:
    ADD_SYNC(dpmsync, 0)
};

//=================================================
//Data types configuration

struct DPMData
{
    QString category;
    int id;
    cv::Rect rect;
    QString attributes;
};

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    QString rosbagfile;
    int frameid;
    bool saveimageflag;
    cv::Mat image;
    QVector<DPMData> dpmdata;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
