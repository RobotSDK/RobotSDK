#ifndef TESTMODULE_H
#define TESTMODULE_H

#include<QTimer>
#include<QTime>

#include<RobotSDK_Global.h>
using namespace RobotSDK;

#ifdef NODE_CLASS
#undef NODE_CLASS
#endif
#define NODE_CLASS RandomGenerator

//#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 0

//#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 2

class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(int, max, 100)
};

class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_VAR(int, offset, 50)
    ADD_INTERNAL_QOBJECT_TRIGGER(QTimer, timer)
    ADD_INTERNAL_DEFAULT_CONNECTION(timer,timeout)
};

class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{
public:
    int result;
};

#endif // TESTMODULE_H
