#ifndef PLANNINGSIM
#define PLANNINGSIM

//=================================================
//Please add headers here:
#include<Planning.h>
#include<QFile>
#include<QTimer>

//=================================================
#include<RobotSDK.h>
namespace RobotSDK_Module
{

//=================================================
//Node configuration

#undef NODE_CLASS
#define NODE_CLASS PlanningSim

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 0

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 1

//=================================================
//Params types configuration

//If you need to refer params type of other node class, please uncomment below and comment its own params type.
NODE_PARAMS_TYPE_REF(Planning)

//=================================================
//Vars types configuration

//If you need to refer vars type of other node class, please uncomment below and comment its own vars type.
//NODE_VARS_TYPE_REF(RefNodeClassName)
class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(QString, orderfile, "order.txt")
    QStringList orders;
    uint orderid;
public:
    ADD_INTERNAL_QOBJECT_TRIGGER(QTimer, timer, 0)
    ADD_INTERNAL_DEFAULT_CONNECTION(timer,timeout)
};

//=================================================
//Data types configuration

//If you need to refer data type of other node class, please uncomment below and comment its own data type.
NODE_DATA_TYPE_REF(Planning)

//=================================================
//You can declare functions here


//=================================================
}

#endif
