#include"Controller.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, Localization)
PORT_DECL(1, Planning)
PORT_DECL(2, Obstacle)

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    NOUNUSEDWARNING;
    SHSpur_init();
    auto vars=NODE_VARS;
    return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    SHSpur_set_vel(params->linear_velocity);
    SHSpur_set_angvel(params->angular_velocity);
    SHSpur_set_pos_GL(0,0,0);

    auto vars=NODE_VARS;
    vars->checkposflag=0;
    vars->checkangflag=0;
    vars->obstacleflag=0;

	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    SHSpur_stop();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(PORT_DATA_SIZE(0)>0)
    {
        auto data=PORT_DATA(0,0);
        SHSpur_set_pos_GL(data->posx,data->posy,data->theta);
    }
    else if(PORT_DATA_SIZE(1)>0)
    {
        auto data=PORT_DATA(1,0);
        QStringList order=data->order.split(",",QString::SkipEmptyParts);
        if(order.at(0)=="Line")
        {
            vars->checkposflag=1;
            vars->checkangflag=0;
            if(order.at(1)=="GL")
            {
                double posx,posy,ang;
                SHSpur_get_pos_GL(&posx,&posy,&ang);
                vars->destposx=order.at(2).toDouble();
                vars->destposy=order.at(3).toDouble();
                vars->destang=atan2(vars->destposy-posy,vars->destposx-posx);
                if(!vars->obstacleflag)
                {
                    SHSpur_line_GL(vars->destposx,vars->destposy,vars->destang);
                }
            }
            else if(order.at(1)=="LC")
            {
                SHSpur_get_pos_GL(&(vars->destposx),&(vars->destposy),&(vars->destang));
                double dx=order.at(2).toDouble();
                double dy=order.at(3).toDouble();
                vars->destposx+=dx;
                vars->destposy+=dy;
                vars->destang+=atan2(dy,dx);
                if(!vars->obstacleflag)
                {
                    SHSpur_line_GL(vars->destposx,vars->destposy,vars->destang);
                }
            }
        }
        else if(order.at(0)=="Spin")
        {
            vars->checkposflag=0;
            vars->checkangflag=1;
            if(order.at(1)=="GL")
            {
                vars->destang=order.at(2).toDouble();
                SHSpur_spin_GL(vars->destang);
            }
            else if(order.at(1)=="LC")
            {
                SHSpur_get_pos_GL(&(vars->destposx),&(vars->destposy),&(vars->destang));
                vars->destang+=order.at(2).toDouble();
                SHSpur_spin_GL(vars->destang);
            }
        }
        else if(order.at(0)=="Stop")
        {
            vars->checkposflag=0;
            vars->checkangflag=0;
            SHSpur_stop();
        }
    }
    else if(PORT_DATA_SIZE(2)>0)
    {
        auto data=PORT_DATA(2,0);
        auto outputdata=NODE_DATA;
        outputdata->state=None;
        bool flag1=vars->obstacleflag;
        bool flag2=data->obstacleflag;
        if(flag1)
        {
            if(!flag2)
            {
                if(vars->checkposflag)
                {
                    SHSpur_line_GL(vars->destposx,vars->destposy,vars->destang);
                }
                vars->obstacleflag=data->obstacleflag;
            }
            else
            {
                outputdata->state=Wait;
            }
        }
        else
        {
            if(flag2)
            {
                if(vars->checkposflag)
                {
                    SHSpur_stop();
                }
                vars->obstacleflag=data->obstacleflag;
                outputdata->state=Wait;
            }
        }
        if(vars->checkposflag)
        {
            if(SHSpur_near_pos_GL(vars->destposx,vars->destposy,vars->pos_error))
            {
                SHSpur_stop();
                vars->checkposflag=0;
                outputdata->state=Finish;
            }
            else
            {
                outputdata->state=Process;
            }
        }
        else if(vars->checkangflag)
        {
            if(SHSpur_near_ang_GL(vars->destang,vars->ang_error))
            {
                SHSpur_stop();
                vars->checkangflag=0;
                outputdata->state=Finish;
            }
            else
            {
                outputdata->state=Process;
            }
        }
        return 1;
    }
    return 0;
}
