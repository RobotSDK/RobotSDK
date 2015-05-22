#include"Planning.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, Controller)
PORT_DECL(1, Localization)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    QFile file(vars->waypointfile);
    if(file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        QTextStream stream(&file);
        vars->waypoint.clear();
        while(!stream.atEnd())
        {
            double x,y;
            stream>>x>>y;
            vars->waypoint.push_back(QPointF(x,y));
        }
        vars->pointid=0;
        vars->tmppointflag=0;
        file.close();
        vars->curposflag=0;
        vars->timer->setSingleShot(1);
        vars->timer->start(1000);
        return 1;
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
	return 1;
}

NODE_FUNC_DEF(bool, generateNewOrder)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;
    if(vars->tmppointflag)
    {
        vars->timer->start(vars->tmpwaittime);
        return 0;
    }
    else
    {
        if(vars->orderlist.size()>0)
        {
            data->order=vars->orderlist.front();
            vars->orderlist.pop_front();
            return 1;
        }
        else
        {
            while(vars->pointid<vars->waypoint.size())
            {
                QPointF diff=vars->curpoint-vars->waypoint.at(vars->pointid);
                double theta=atan2(diff.y(),diff.x());
                double distance=sqrt(diff.x()*diff.x()+diff.y()*diff.y());
                if(distance>params->pos_error)
                {
                    vars->orderlist.push_back(QString("Spin,GL,%1").arg(theta));
                    vars->orderlist.push_back(QString("Line,LC,%1,0").arg(distance));
                    vars->orderlist.push_back(QString("Stop"));
                    data->order=vars->orderlist.front();
                    vars->orderlist.pop_front();
                    return 1;
                }
                vars->pointid++;
            }
        }
    }
    return 0;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(IS_INTERNAL_TRIGGER)
    {
        if(vars->curposflag)
        {
            return NODE_FUNC(generateNewOrder);
        }
        else
        {
            vars->timer->start(1000);
        }
    }
    else if(PORT_DATA_SIZE(0)>0)
    {

            auto inputdata=PORT_DATA(0,0);
            switch(inputdata->state)
            {
            case None:
                if(vars->curposflag)
                {
                    vars->tmppointflag=0;
                    return NODE_FUNC(generateNewOrder);
                }
                break;
            case Process:
                vars->tmppointflag=0;
                vars->timer->stop();
                break;
            case Finish:
                vars->tmppointflag=0;
                return NODE_FUNC(generateNewOrder);
            case Wait:
                if(!vars->tmppointflag)
                {
                    vars->tmppointflag=1;
                    vars->timer->start(vars->tmpwaittime);
                }
                break;
            default:
                break;
            }
    }
    else if(PORT_DATA_SIZE(1)>0)
    {
        auto inputdata=PORT_DATA(1,0);
        vars->curpoint=QPointF(inputdata->posx,inputdata->posy);
        vars->curtheta=inputdata->theta;
        vars->curposflag=1;
    }
    return 0;
}
