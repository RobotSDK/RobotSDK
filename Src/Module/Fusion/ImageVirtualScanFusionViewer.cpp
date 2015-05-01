#include"ImageVirtualScanFusionViewer.h"

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraVirtualScanFusion)

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

    vars->colortable.resize(256);
    int i;
    for(i=0;i<256;i++)
    {
        vars->colortable[i]=qRgb(i,i,i);
    }
    return 1;
}

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    vars->viewer->setText("Open");
    cv::Mat greymat(1,256,CV_8UC1);
    uint i,n=params->colormapsegments;
    for(i=0;i<256/n;i++)
    {
        uint j;
        for(j=0;j<n;j++)
        {
            greymat.at<uchar>(i+j*256/n)=n*i+j;
        }
    }
    cv::applyColorMap(greymat,vars->colormap,params->colormapid);
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    auto vars=NODE_VARS;
    vars->viewer->setText("Close");
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=PORT_DATA(0,0);

    vars->tabwidget->setTabText(0,QString("%1 ~ %2").arg(data->timestamp.toString("HH:mm:ss:zzz")).arg(data->virtualscantimestamp.toString("HH:mm:ss:zzz")));

    cv::Mat image=data->cvimage.clone();

    if(image.type()==CV_8UC3)
    {
        if(data->clusternum==0)
        {
            QMultiMap< uint, QPair< QPoint, QPoint > >::const_iterator stixeliter;
            for(stixeliter=data->stixel.begin();stixeliter!=data->stixel.end();stixeliter++)
            {
                cv::Point2i start=cv::Point2i(stixeliter.value().first.x(),stixeliter.value().first.y());
                cv::Point2i end=cv::Point2i(stixeliter.value().second.x(),stixeliter.value().second.y());
                cv::circle(image,start,1,cv::Scalar(255,0,0));
                cv::circle(image,end,1,cv::Scalar(255,0,0));
                cv::line(image,start,end,cv::Scalar(255,0,0));
            }
        }
        else
        {
            uint i;
            for(i=0;i<data->clusternum;i++)
            {
                QList< QPair< QPoint, QPoint > > pointslist=data->stixel.values(i);
                cv::Vec3b color;
                if(i==0)
                {
                    color[0]=128;
                    color[1]=128;
                    color[2]=128;
                }
                else
                {
                    int colorid=int(255.0*i/data->clusternum+0.5);
                    if(vars->colormap.type()==CV_8UC3)
                    {
                        color=vars->colormap.at<cv::Vec3b>(colorid);
                    }
                    else if(vars->colormap.type()==CV_8UC1)
                    {
                        color[0]=vars->colormap.at<uchar>(colorid);
                        color[1]=vars->colormap.at<uchar>(colorid);
                        color[2]=vars->colormap.at<uchar>(colorid);
                    }
                }
                uint j,m=pointslist.size();
                for(j=0;j<m;j++)
                {
                    cv::Point2i start=cv::Point2i(pointslist.at(j).first.x(),pointslist.at(j).first.y());
                    cv::Point2i end=cv::Point2i(pointslist.at(j).second.x(),pointslist.at(j).second.y());
                    cv::circle(image,start,1,cv::Scalar(color[0],color[1],color[2]));
                    cv::circle(image,end,1,cv::Scalar(color[0],color[1],color[2]));
                    cv::line(image,start,end,cv::Scalar(color[0],color[1],color[2]));
                }
            }
        }
        QImage img(image.data,image.cols,image.rows,image.step,QImage::Format_RGB888);
        vars->viewer->setPixmap(QPixmap::fromImage(img));
        vars->viewer->resize(img.size());
    }
    else if(image.type()==CV_8UC1)
    {
        QImage img(image.data,image.cols,image.rows,image.step,QImage::Format_Indexed8);
        img.setColorTable(vars->colortable);
        vars->viewer->setPixmap(QPixmap::fromImage(img));
        vars->viewer->resize(img.size());
    }
    else
    {
        vars->viewer->setText("Not Support");
        return 0;
    }
    return 1;
}
