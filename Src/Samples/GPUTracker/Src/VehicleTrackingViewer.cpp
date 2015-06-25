#include"VehicleTrackingViewer.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, ObstacleMapGenerator)
PORT_DECL(1, VehicleTracker)

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->layout->addWidget(vars->viewer);
    vars->widget->setLayout(vars->layout);
    vars->setNodeGUIThreadFlag(1);
	return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->clear();
    SYNC_CLEAR(vars->sync);
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->clear();
    SYNC_CLEAR(vars->sync);
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(SYNC_START(vars->sync))
    {
        auto mapdata=SYNC_DATA(vars->sync,0);
        auto mapparam=SYNC_PARAMS(vars->sync,0);
        auto detection=SYNC_DATA(vars->sync,1);
        QImage img(mapdata->map.data,mapdata->map.cols,mapdata->map.rows,mapdata->map.step,QImage::Format_RGB888);
        vars->viewer->setPixmap(img);
        int i,n=detection->objectid.size();
        for(i=0;i<n;i++)
        {
            vars->viewer->addTrackingResult(detection->objectstate[i].x/mapparam->gridsize,detection->objectstate[i].y/mapparam->gridsize,detection->objectstate[i].theta
                                            ,detection->objectstate[i].width/mapparam->gridsize,detection->objectstate[i].length/mapparam->gridsize);
        }
    }
    return 0;
}
