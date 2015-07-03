#include"VehicleDetector.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VirtualScanGlobalizer)

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

NODE_FUNC_DEF(void, drawImage)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto vscan=PORT_DATA(0,0);
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
    vars->detector->setPixmap(img);
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto vscan=PORT_DATA(0,0);
    auto data=NODE_DATA;
    if(vars->timestamp.isNull())
    {
        vars->timestamp=vscan->timestamp;
        NODE_FUNC(drawImage);
        return 0;
    }
    else
    {
        data->timestamp=vars->timestamp;
        QVector<QLineF> detection=vars->detector->getDetection();
        int i,n=detection.size();
        data->objectids.resize(n);
        data->objectstates.resize(n);
        int center=params->imagesize/2;
        float ratio=float(2*params->maxrange+1)/float(params->imagesize);
        for(i=0;i<n;i++)
        {
            data->objectids[i]=vars->idcount++;
            data->objectstates[i].x=(center-detection[i].p1().y())*ratio;
            data->objectstates[i].y=(center-detection[i].p1().x())*ratio;
            float dx=(detection[i].p1().y()-detection[i].p2().y())*ratio;
            float dy=(detection[i].p1().x()-detection[i].p2().x())*ratio;
            data->objectstates[i].theta=atan2(dy,dx);
            data->objectstates[i].v=10;
            data->objectstates[i].width=2;
            data->objectstates[i].length=3;
        }
        vars->timestamp=vscan->timestamp;
        NODE_FUNC(drawImage);
        return 1;
    }
}
