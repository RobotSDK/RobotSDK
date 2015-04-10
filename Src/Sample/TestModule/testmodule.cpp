#include "testmodule.h"
using namespace RobotSDK;

USE_DEFAULT_NODE

NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    auto vars=NODE_VARS;
    vars->timer->start(1000);
    return 1;
}

NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    return 1;
}

NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    return 1;
}

NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;
    data->result=random()%params->max+vars->offset;
    uint out=data->result%2;
    if(out==0)
    {
        data->setOutputPortFilterFlag(QList<bool>()<<0<<1);
    }
    else
    {
        data->setOutputPortFilterFlag(QList<bool>()<<1<<0);
    }
    return 1;
}
