#ifndef TESTMODULE_H
#define TESTMODULE_H

#include<QTimer>
#include<QLabel>

#include<RobotSDK_Global.h>

#undef NODE_CLASS
#define NODE_CLASS RandomGenerator

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 0

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 2

class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(int, max, 100)
    ADD_PARAM(int, interval, 1000)
};

class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(int, offset, 0)
    ADD_VAR(QString, format, QString("HH:mm:ss:zzz"))
    ADD_INTERNAL_QOBJECT_TRIGGER(QTimer, timer,0)
    ADD_INTERNAL_DEFAULT_CONNECTION(timer,timeout)
    ADD_QWIDGET(QLabel, number)
    ADD_QLAYOUT(QHBoxLayout, layout)
};

class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    int result;
};

#endif // TESTMODULE_H
