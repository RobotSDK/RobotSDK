#ifndef NUMBERVIEWER
#define NUMBERVIEWER

#include"randomgenerator.h"

#include<RobotSDK_Global.h>
using namespace RobotSDK;

#undef NODE_CLASS
#define NODE_CLASS NumberViewer

#undef INPUT_PORT_NUM
#define INPUT_PORT_NUM 1

#undef OUTPUT_PORT_NUM
#define OUTPUT_PORT_NUM 0

PORT_DECL(0, RandomGenerator)

class NODE_PARAMS_TYPE : public NODE_PARAMS_BASE_TYPE
{
public:
    ADD_PARAM(QString, format, QString("HH:mm:ss"))
};

class NODE_VARS_TYPE : public NODE_VARS_BASE_TYPE
{
public:
    ADD_QWIDGET(QLabel, number)
    ADD_QLAYOUT(QHBoxLayout, layout)
};

class NODE_DATA_TYPE : public NODE_DATA_BASE_TYPE
{

};

#endif // NUMBERVIEWER

