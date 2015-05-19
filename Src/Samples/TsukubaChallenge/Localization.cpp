#include"Localization.h"
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
    if(vars->odmsub==NULL)
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
    if(vars->odmsub==NULL)
    {
        return 0;
    }
    vars->odmsub->resetDestinationFrame(vars->destinationframe);
    vars->odmsub->resetOriginalFrame(vars->originalframe);
    vars->odmsub->startReceiveSlot();
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->odmsub->stopReceiveSlot();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;

    tf::StampedTransform rostransform;
    if(!(vars->odmsub->getTF(rostransform)))
    {
        return 0;
    }
    int msec=(rostransform.stamp_.sec)%(24*60*60)*1000+(rostransform.stamp_.nsec)/1000000;
    data->timestamp=QTime::fromMSecsSinceStartOfDay(msec);

    tf::Vector3 translation=rostransform.getOrigin();
    data->posx=translation.getX();
    data->posy=translation.getY();

    tf::Matrix3x3 rotation(rostransform.getRotation());
    tf::Vector3 forward(1.0,0.0,0.0);
    forward=rotation*forward;
    data->theta=atan2(forward.getY(),forward.getX());

	return 1;
}
