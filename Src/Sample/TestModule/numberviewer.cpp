#include"numberviewer.h"

USE_DEFAULT_NODE

NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    auto vars=NODE_VARS;
    vars->widget->setLayout(vars->layout);
    vars->layout->addWidget(vars->number);
    vars->number->setAlignment(Qt::AlignCenter);
    return 1;
}

NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;    
    auto data=PORT_DATA(0,0);
    vars->number->setText(QString("%1::%2\n%3\n%4").arg(params->nodeclass).arg(params->nodename).arg(data->result).arg(data->timestamp.toString(params->format)));
    return 1;
}
