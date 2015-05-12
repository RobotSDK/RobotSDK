#include"DPMDetector.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Original node functions

//If you don't need initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    if(vars->dpmsub==NULL)
    {
        return 0;
    }
    return 1;
}

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    if(vars->dpmsub==NULL)
    {
        return 0;
    }
    vars->dpmsub->resetTopic(vars->topic,vars->queuesize);
    vars->dpmsub->resetQueryInterval(vars->queryinterval);
    vars->dpmsub->startReceiveSlot();
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    vars->dpmsub->stopReceiveSlot();
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;

    outputdata->rosdetection=vars->dpmsub->getMessage();
    if(outputdata->rosdetection==NULL)
    {
        return 0;
    }
    int msec=(outputdata->rosdetection->header.stamp.sec)%(24*60*60)*1000+(outputdata->rosdetection->header.stamp.nsec)/1000000;
    outputdata->timestamp=QTime::fromMSecsSinceStartOfDay(msec);

    outputdata->detection.resize(outputdata->rosdetection->car_num);
    int i;
    for(i=0;i<outputdata->rosdetection->car_num;i++)
    {
        outputdata->detection[i].x=outputdata->rosdetection->corner_point[i*4];
        outputdata->detection[i].y=outputdata->rosdetection->corner_point[i*4+1];
        outputdata->detection[i].width=outputdata->rosdetection->corner_point[i*4+2];
        outputdata->detection[i].height=outputdata->rosdetection->corner_point[i*4+3];
    }
    return 1;
}
