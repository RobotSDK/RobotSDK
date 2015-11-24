#include"VirtualScanGenerator.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VelodyneSensor)

//=================================================
//Original node functions

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;
    auto inputdata=PORT_DATA(0,0);

    data->timestamp=inputdata->timestamp;
    data->extrinsicmat=inputdata->extrinsicmat.clone();
    data->rospoints=inputdata->rospoints;
    double PI=3.141592654;
    vars->virtualscan.velodyne=inputdata->rospoints;
    vars->virtualscan.calculateVirtualScans(params->beamnum,params->heightstep,params->minfloor,params->maxceiling,params->obstacleminheight,params->maxbackdistance,params->rotation*PI/180.0,params->minrange);
    vars->virtualscan.getVirtualScan(params->slope*PI/180.0,params->maxfloor,params->minceiling,params->passheight,data->virtualscan);
    data->minheights=vars->virtualscan.minheights;
    data->maxheights=vars->virtualscan.maxheights;
    data->labels.fill(0,params->beamnum);
    data->clusternum=0;
    data->clusters.clear();
    return 1;
}
