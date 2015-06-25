#include"VehicleDetector.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, ObstacleMapGenerator)

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->layout->addWidget(vars->detector);
    vars->widget->setLayout(vars->layout);
    vars->setNodeGUIThreadFlag(1);
	return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->detector->clear();
    vars->timestamp=QTime();
    vars->idcount=0;
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->detector->clear();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto mapdata=PORT_DATA(0,0);
    auto mapparam=PORT_PARAMS(0,0);
    auto data=NODE_DATA;

    if(vars->timestamp.isNull())
    {
        vars->timestamp=mapdata->timestamp;
        QImage img(mapdata->map.data,mapdata->map.cols,mapdata->map.rows,mapdata->map.step,QImage::Format_RGB888);
        vars->detector->setPixmap(img);
        return 0;
    }
    else
    {
        data->timestamp=vars->timestamp;
        QVector<QLineF> detection=vars->detector->getDetection();
        int i,n=detection.size();
        data->objectids.resize(n);
        data->objectstates.resize(n);
        int center=mapdata->map.rows/2;
        for(i=0;i<n;i++)
        {
            data->objectids[i]=vars->idcount++;
            data->objectstates[i].x=(center-detection[i].p1().y())*mapparam->gridsize;
            data->objectstates[i].y=(center-detection[i].p1().x())*mapparam->gridsize;
            double dx=(detection[i].p1().y()-detection[i].p2().y())*mapparam->gridsize;
            double dy=(detection[i].p1().x()-detection[i].p2().x())*mapparam->gridsize;
            data->objectstates[i].theta=atan2(dy,dx);
            data->objectstates[i].v=10;
            data->objectstates[i].width=2;
            data->objectstates[i].length=3;
        }
        vars->timestamp=mapdata->timestamp;
        QImage img(mapdata->map.data,mapdata->map.cols,mapdata->map.rows,mapdata->map.step,QImage::Format_RGB888);
        vars->detector->setPixmap(img);
        return 1;
    }
}
