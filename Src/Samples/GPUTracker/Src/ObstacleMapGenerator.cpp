#include"ObstacleMapGenerator.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VirtualScanGenerator)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    vars->mapsize=(params->maprange/params->gridsize)*2+1;
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;
    auto inputdata=PORT_DATA(0,0);
    data->timestamp=inputdata->timestamp;
    data->map=cv::Mat(vars->mapsize,vars->mapsize,CV_8UC3);
    data->mapdata=cv::Mat(vars->mapsize,vars->mapsize,CV_32F);
    cudaObstacleMapGenerator(inputdata->virtualscan.size(),inputdata->virtualscan.data()
                          ,vars->mapsize,params->gridsize,params->obstaclefactor,data->map.data,(float *)(data->mapdata.data));
	return 1;
}
