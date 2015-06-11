#include"DPMFakeReceiver.h"
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

    sensor_msgs::ImageConstPtr rosimage=vars->dpmsub->getMessage();
    int msec=(rosimage->header.stamp.sec)%(24*60*60)*1000+(rosimage->header.stamp.nsec)/1000000;
    data->timestamp=QTime::fromMSecsSinceStartOfDay(msec);

    data->id=vars->
    data->detection.clear();
    return 1;

	return 1;
}
