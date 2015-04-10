#include "randomgenerator.h"

USE_DEFAULT_NODE

NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    auto vars=NODE_VARS;
    vars->widget->setLayout(vars->layout);
    vars->layout->addWidget(vars->number);
    vars->number->setAlignment(Qt::AlignCenter);
    vars->timer->setTimerType(Qt::PreciseTimer);
    return 1;
}

NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    auto vars=NODE_VARS;
    auto params=NODE_PARAMS;
    vars->timer->start(params->interval);
    return 1;
}

NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    auto vars=NODE_VARS;
    vars->timer->stop();
    return 1;
}

NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;

    data->result=qrand()%params->max+vars->offset;
    data->timestamp=QTime::currentTime();

    vars->number->setText(QString("%1::%2\n%3\n%4").arg(params->nodeclass).arg(params->nodename).arg(data->result).arg(data->timestamp.toString(vars->format)));

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
