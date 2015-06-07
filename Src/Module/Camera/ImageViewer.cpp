#include"ImageViewer.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraSensor)

//=================================================
//Original node functions

//If you don't need initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->setAlignment(Qt::AlignCenter);
    vars->scrollarea->setWidget(vars->viewer);
    vars->tabwidget->addTab(vars->scrollarea,"TimeStamp");
    vars->layout->addWidget(vars->tabwidget);
    vars->widget->setLayout(vars->layout);

    vars->colortable.clear();
    uint i;
    for(i=0;i<256;i++)
    {
        vars->colortable.push_back(qRgb(i,i,i));
    }

    vars->setNodeGUIThreadFlag(1);
    return 1;
}

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->setText("Open");
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->setText("Close");
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=PORT_DATA(0,0);

    vars->tabwidget->setTabText(0,data->timestamp.toString("HH:mm:ss:zzz"));
    qDebug()<<data->timestamp;

    cv::Mat image=data->cvimage;

    if(image.type()==CV_8UC3)
    {
        QImage img(image.data,image.cols,image.rows,image.step,QImage::Format_RGB888);
        vars->viewer->setPixmap(QPixmap::fromImage(img));
        vars->viewer->resize(img.size());
    }
    else if(image.type()==CV_8UC1)
    {
        QImage img(image.data,image.cols,image.rows,image.step,QImage::Format_Indexed8);
        img.setColorCount(256);
        img.setColorTable(vars->colortable);
        vars->viewer->setPixmap(QPixmap::fromImage(img));
        vars->viewer->resize(img.size());
    }
    else if(image.type()==CV_16UC1)
    {
        QImage img(image.cols,image.rows,QImage::Format_Indexed8);
        img.fill(0);
        img.setColorCount(256);
        img.setColorTable(vars->colortable);
        uint i,j;
        ushort factor=25;
        for(i=0;i<image.cols;i++)
        {
            for(j=0;j<image.rows;j++)
            {
                if(image.at<ushort>(j,i)/factor>=256)
                {
                    img.setPixel(i,j,255);
                }
                else
                {
                    img.setPixel(i,j,image.at<ushort>(j,i)/factor);
                }
            }
        }
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
