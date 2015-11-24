#include"VehicleTrackingViewer.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VehicleTracker)
PORT_DECL(1, VirtualScanGlobalizer)

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->layout->addWidget(vars->viewer);
    vars->layout->addWidget(vars->list);
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
    vars->list->clear();
    SYNC_CLEAR(vars->sync);
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->clear();
    vars->list->clear();
    SYNC_CLEAR(vars->sync);
	return 1;
}

NODE_FUNC_DEF(void, drawImage)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;

    auto vscan=SYNC_DATA(vars->sync,1);
    QImage img(params->imagesize,params->imagesize,QImage::Format_RGB888);
    img.fill(QColor(255,255,255));
    QPainter painter;
    painter.begin(&img);

    float ratio=float(params->imagesize)/float(2*params->maxrange+1);
    float PI=3.141592654;
    int i,beamnum=vscan->virtualscan.size();
    float density=2*PI/beamnum;

    painter.setPen(QColor(0,0,0));
    QPoint center(params->imagesize/2,params->imagesize/2);
    for(i=params->gridsize;i<=params->maxrange;i+=params->gridsize)
    {
        painter.drawEllipse(center,int(i*ratio),int(i*ratio));
    }
    painter.setPen(QColor(255,0,0));
    for(i=0;i<beamnum;i++)
    {
        float theta=i*density-PI;
        int y=int((params->maxrange-vscan->virtualscan[i]*cos(theta))*ratio+0.5);
        int x=int((params->maxrange-vscan->virtualscan[i]*sin(theta))*ratio+0.5);
        painter.drawEllipse(QPoint(x,y),1,1);
    }
    vars->viewer->setPixmap(img);
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto params=NODE_PARAMS;
    if(SYNC_START(vars->sync))
    {
        NODE_FUNC(drawImage);
        auto detection=SYNC_DATA(vars->sync,0);
        int i,n=detection->objectid.size();
        float ratio=float(params->imagesize)/float(2*params->maxrange+1);
        vars->list->clear();
        for(i=0;i<n;i++)
        {
            vars->viewer->addTrackingResult(detection->objectstate[i].x*ratio,detection->objectstate[i].y*ratio,detection->objectstate[i].theta
                                            ,detection->objectstate[i].width*ratio,detection->objectstate[i].length*ratio);
            vars->list->addItem(QString("ID=%1, Weight=%2, X=%3, Y=%4, A=%5, V=%6, W=%7, L=%8").arg(i).arg(detection->weights[i])
                                .arg(detection->objectstate[i].x).arg(detection->objectstate[i].y).arg(detection->objectstate[i].theta)
                                .arg(detection->objectstate[i].v).arg(detection->objectstate[i].width).arg(detection->objectstate[i].length));
        }
    }
	return 1;
}
