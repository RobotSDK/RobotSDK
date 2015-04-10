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
    auto vars=NODE_VARS;
    auto params=NODE_PARAMS;
    auto data=PORT_DATA(0,0);
    vars->number->setText(QString("%1\n%2").arg(data->result).arg(data->timestamp.toString(params->format)));
    return 1;
}
