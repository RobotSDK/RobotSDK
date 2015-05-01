#include"ImagePointCloudFusionViewer.h"

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraVelodyneFusion)

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

    cv::Mat greymat=cv::Mat(1,256,CV_8UC1);
    int i;
    for(i=0;i<256;i++)
    {
        greymat.at<uchar>(i)=i;
    }
    cv::applyColorMap(greymat,vars->colormap,params->colormapid);

    vars->viewer->setText("Open");
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

    vars->tabwidget->setTabText(0,QString("%1 ~ %2").arg(data->timestamp.toString("HH:mm:ss:zzz")).arg(data->velodynetimestamp.toString("HH:mm:ss:zzz")));

    cv::Mat image=data->cvimage.clone();

    if(image.type()==CV_8UC3)
    {
        QMap< QPair<int,int>, double >::const_iterator rangeiter;
        for(rangeiter=data->ranges.begin();rangeiter!=data->ranges.end();rangeiter++)
        {
            if(rangeiter.value()>params->minrange&&rangeiter.value()<=params->maxrange)
            {
                int colorid=int((rangeiter.value()-params->minrange)/(params->maxrange-params->minrange)*255.0+0.5);
                if(vars->colormap.type()==CV_8UC3)
                {
                    cv::Vec3b color=vars->colormap.at<cv::Vec3b>(colorid);
                    cv::circle(image,cv::Point2i(rangeiter.key().first,rangeiter.key().second),1,cv::Scalar(color.val[0],color.val[1],color.val[2]));
                }
                else if(vars->colormap.type()==CV_8UC1)
                {
                    uchar color=vars->colormap.at<uchar>(colorid);
                    cv::circle(image,cv::Point2i(rangeiter.key().first,rangeiter.key().second),1,cv::Scalar(color,color,color));
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
