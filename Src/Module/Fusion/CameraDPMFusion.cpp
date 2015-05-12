#include"CameraDPMFusion.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraSensor)
PORT_DECL(1, DPMDetector)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->dpmsync);
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->dpmsync);
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    bool flag=SYNC_START(vars->dpmsync);
    if(flag)
    {
        auto imagedata=SYNC_DATA(vars->dpmsync,0);
        auto dpmdata=SYNC_DATA(vars->dpmsync,1);
        auto outputdata=NODE_DATA;

        outputdata->timestamp=imagedata->timestamp;

        outputdata->extrinsicmat=imagedata->extrinsicmat.clone();

        uint i,n=dpmdata->detection.size();
        for(i=0;i<n;i++)
        {

        }
    }
	return 1;
}
