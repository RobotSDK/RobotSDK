#include"ObstacleMapGlobalizer.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, ObstacleMapGenerator)
PORT_DECL(1, NDTLocalizer)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->sync.clear();
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->sync.clear();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(SYNC_START(vars->sync))
    {
        auto mapparams=SYNC_PARAMS(vars->sync,0);
        auto mapdata=SYNC_DATA(vars->sync,0);
        auto localization=SYNC_DATA(vars->sync,1);
        auto data=NODE_DATA;
        data->timestamp=mapdata->timestamp;
        data->gridsize=mapparams->gridsize;
        data->maprange=mapparams->maprange;
        data->radius=mapparams->obstaclefactor*data->gridsize;
        data->map=mapdata->map;
        data->mapdata=mapdata->mapdata;
        data->transform=localization->cvtransform;
        return 1;
    }
    else
    {
        return 0;
    }
}
