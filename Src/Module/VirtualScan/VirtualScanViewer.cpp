#include"VirtualScanViewer.h"
using namespace RobotSDK_Module;

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
    NOUNUSEDWARNING
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
    NOUNUSEDWARNING
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    vars->viewer->setText("Open");
    cv::Mat grayscale(1,256,CV_8UC1);
    uint i,n=params->colormapsegments;
    for(i=0;i<256/n;i++)
    {
        uint j;
        for(j=0;j<n;j++)
        {
            grayscale.at<uchar>(i+j*256/n)=n*i+j;
        }
    }
    cv::applyColorMap(grayscale,vars->colortable,cv::COLORMAP_RAINBOW);
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    vars->viewer->setText("Closed");
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING
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
    uint i,beamnum=data->virtualscan.size();
    double density=2*PI/beamnum;

    painter.setPen(QColor(0,0,0));
    QPoint center(params->imagesize/2,params->imagesize/2);
    for(i=params->gridsize;i<=params->maxrange;i+=params->gridsize)
    {
        painter.drawEllipse(center,int(i*ratio),int(i*ratio));
    }
    if(data->clusternum==0)
    {
        painter.setPen(QColor(255,0,0));
        for(i=0;i<beamnum;i++)
        {
            double theta=i*density+PI/2;
            int x=int((params->maxrange+data->virtualscan[i]*cos(theta))*ratio+0.5);
            int y=int((params->maxrange+data->virtualscan[i]*sin(theta))*ratio+0.5);
            x=params->imagesize-x;
            painter.drawEllipse(QPoint(x,y),1,1);
        }
    }
    else
    {
        for(i=0;i<=data->clusternum;i++)
        {
            if(i==0)
            {
                painter.setPen(QColor(128,128,128));
            }
            else
            {
                int colorid=int(255.0*i/data->clusternum+0.5);
                cv::Vec3b color=vars->colortable.at<cv::Vec3b>(colorid);
                painter.setPen(QColor(color[0],color[1],color[2]));
            }
            QList<uint> pointsid=data->clusters.values(i);
            uint j,m=pointsid.size();
            for(j=0;j<m;j++)
            {
                double theta=pointsid[j]*density+PI/2;
                int x=int((params->maxrange+data->virtualscan[pointsid[j]]*cos(theta))*ratio+0.5);
                int y=int((params->maxrange+data->virtualscan[pointsid[j]]*sin(theta))*ratio+0.5);
                x=params->imagesize-x;
                painter.drawEllipse(QPoint(x,y),1,1);
            }
        }
    }
    painter.end();

    vars->viewer->setPixmap(QPixmap::fromImage(image));
    vars->viewer->resize(image.size());
    return 1;
}
