#include"DPMReceiver.h"
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
    vars->dpmsub->resetTopic(vars->rostopic,vars->rosqueuesize);
    vars->dpmsub->resetQueryInterval(vars->rosqueryinterval);
    vars->dpmsub->startReceiveSlot();
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->dpmsub->stopReceiveSlot();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;

    dpm::ImageObjectsConstPtr rosdetection=vars->dpmsub->getMessage();
    int msec=(rosdetection->header.stamp.sec)%(24*60*60)*1000+(rosdetection->header.stamp.nsec)/1000000;
    data->timestamp=QTime::fromMSecsSinceStartOfDay(msec);

    data->detection.resize(rosdetection->car_num);
    data->id.resize(rosdetection->car_num);
    int i;
    for(i=0;i<rosdetection->car_num;i++)
    {
        data->detection[i].x=rosdetection->corner_point[i*4];
        data->detection[i].y=rosdetection->corner_point[i*4+1];
        data->detection[i].width=rosdetection->corner_point[i*4+2];
        data->detection[i].height=rosdetection->corner_point[i*4+3];
        if(vars->idflag)
        {
            data->id[i]=rosdetection->car_type[i];
        }
        else
        {
            data->id[i]=vars->curid++;
        }
    }
	return 1;
}
