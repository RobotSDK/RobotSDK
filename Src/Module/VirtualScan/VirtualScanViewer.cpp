#include"VirtualScanViewer.h"

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VirtualScanGenerator)

//=================================================
//Original node functions

//If you don't need initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    auto vars=NODE_VARS;
    vars->viewer->setAlignment(Qt::AlignCenter);
    vars->scrollarea->setWidget(vars->viewer);
    vars->tabwidget->addTab(vars->scrollarea,"TimeStamp");
    vars->layout->addWidget(vars->tabwidget);
    vars->widget->setLayout(vars->layout);
    vars->setNodeGUIThreadFlag(1);
    return 1;
}

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    auto vars=NODE_VARS;
    vars->viewer->setText("Open");
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    auto vars=NODE_VARS;
    vars->viewer->setText("Closed");
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=PORT_DATA(0,0);

    vars->tabwidget->setTabText(0,data->timestamp.toString("HH:mm:ss:zzz"));

    QImage image(params->imagesize,params->imagesize,QImage::Format_RGB888);
    image.fill(QColor(255,255,255));
    QPainter painter;
    painter.begin(&image);

    double ratio=double(params->imagesize)/double(2*params->maxrange+1);
    double PI=3.141592654;
    int i,beamnum=data->virtualscan.size();
    double density=2*PI/beamnum;

    painter.setPen(QColor(0,0,0));
    QPoint center(params->imagesize/2,params->imagesize/2);
    for(i=params->gridsize;i<=params->maxrange;i+=params->gridsize)
    {
        painter.drawEllipse(center,i*ratio,i*ratio);
    }
    painter.setPen(QColor(255,0,0));
    for(i=0;i<beamnum;i++)
    {
        double theta=i*density+PI/2;
        int x=int((params->maxrange+data->virtualscan[i]*cos(theta))*ratio+0.5);
        int y=int((params->maxrange+data->virtualscan[i]*sin(theta))*ratio+0.5);
        x=params->imagesize-x;
        painter.drawEllipse(QPoint(x,y),1,1);
    }
    painter.end();

    vars->viewer->setPixmap(QPixmap::fromImage(image));
    vars->viewer->resize(image.size());
    return 1;
}
