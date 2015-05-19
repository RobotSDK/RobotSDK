#include"PlanningSim.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    QFile file(vars->orderfile);
    if(file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        vars->timer->setSingleShot(1);
        vars->orders=QString(file.readAll()).split("\n",QString::SkipEmptyParts);
        file.close();
        if(vars->orders.size()>0)
        {
            vars->orderid=0;
            vars->timer->start(1000);
            return 1;
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->timer->stop();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;
    data->order=vars->orders[vars->orderid++];
    if(vars->orderid<vars->orders.size())
    {
        vars->timer->start(vars->orders[vars->orderid++].toInt());
    }
	return 1;
}
