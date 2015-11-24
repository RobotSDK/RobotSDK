#ifndef PLANNING
#define PLANNING

//=================================================
//Please add headers here:
#include<Controller.h>
#include<QPointF>
#include<QPair>
#include<QFile>
#include<QTextStream>
#include<QTimer>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS Planning

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
    ADD_PARAM(double, pos_error, 0.2)
    ADD_PARAM(double, ang_error, 0.04)
};

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(double, tmprot, 90)
    ADD_VAR(double, tmpmove, 1)
    ADD_VAR(int, tmpwaittime, 5000)
public:
    ADD_INTERNAL_QOBJECT_TRIGGER(QTimer,timer,0)
    ADD_INTERNAL_DEFAULT_CONNECTION(timer,timeout)
public:
    ADD_VAR(QString, waypointfile, "waypoint.txt")
    QList<QPointF> waypoint;
    uint pointid;
    bool tmppointflag;
    QPointF tmppoint;
    QList<QString> orderlist;
public:
    bool curposflag;
    QPointF curpoint;
    double curtheta;
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
//NODE_DATA_TYPE_REF(RefNodeClassName)
class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    QString order;
};

//=================================================
//You can declare functions here


//=================================================
}

#endif
