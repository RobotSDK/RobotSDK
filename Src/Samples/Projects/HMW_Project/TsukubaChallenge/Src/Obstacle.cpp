#include"Obstacle.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(vars->urgsub==NULL)
    {
        return 0;
    }
	return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(vars->urgsub==NULL)
    {
        return 0;
    }
    vars->urgsub->resetTopic(vars->topic,vars->queuesize);
    vars->urgsub->resetQueryInterval(vars->queryinterval);
    vars->urgsub->startReceiveSlot();
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->urgsub->stopReceiveSlot();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;

    auto rosscan=vars->urgsub->getMessage();
    if(rosscan==NULL)
    {
        return 0;
    }

    int msec=(rosscan->header.stamp.sec)%(24*60*60)*1000+(rosscan->header.stamp.nsec)/1000000;
    data->timestamp=QTime::fromMSecsSinceStartOfDay(msec);

    data->obstacleflag=0;
    uint i,n=rosscan->ranges.size();
    for(i=0;i<n;i++)
    {
        if(rosscan->ranges[i]>0)
        {
            double theta=rosscan->angle_min+i*rosscan->angle_increment+1.570796327;
            double x=rosscan->ranges[i]*cos(theta);
            double y=rosscan->ranges[i]*sin(theta);
            if(x>=-params->width/2&&x<=params->width/2)
            {
                if(y>=0&&y<=params->height)
                {
                    data->obstacleflag=1;
                    break;
                }
            }
        }
    }
	return 1;
}
