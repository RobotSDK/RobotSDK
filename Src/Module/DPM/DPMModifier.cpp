#include"DPMModifier.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, CameraSensor)
PORT_DECL(1, DPMDetector)

//=================================================
//Original node functions

//If you don't need initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    vars->layout->addWidget(vars->tabwidget);
    vars->tabwidget->addTab(vars->viewer,"TimeStamp");
    vars->layout->addLayout(vars->buttonlayout);
    vars->buttonlayout->addStretch();
    vars->buttonlayout->addWidget(vars->apply);
    vars->widget->setLayout(vars->layout);
    vars->setNodeGUIThreadFlag(1);
    return 1;
}

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    NOUNUSEDWARNING
    auto vars=NODE_VARS;
    SYNC_CLEAR(vars->dpmsync);
    return 1;
}

//If you don't need manually close node, you can delete this code segment
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
    if(IS_INTERNAL_TRIGGER)
    {
        auto data=NODE_DATA;
        data->cvimage=vars->image;
        QVector<QRectF>  rects=vars->viewer->getRects();
        uint i,n=rects.size();
        data->detection.resize(n);
        for(i=0;i<n;i++)
        {
            data->detection[i].x=rects[i].x();
            data->detection[i].y=rects[i].y();
            data->detection[i].width=rects[i].width();
            data->detection[i].height=rects[i].height();
        }
        return 1;
    }
    else
    {
        bool flag=SYNC_START(vars->dpmsync);
        if(flag)
        {
            auto imagedata=SYNC_DATA(vars->dpmsync,0);
            auto dpmdata=SYNC_DATA(vars->dpmsync,1);
            vars->tabwidget->setTabText(0,QString("%1 ~ %2").arg(imagedata->timestamp.toString("HH:mm:ss:zzz")).arg(dpmdata->timestamp.toString("HH:mm:ss:zzz")));

            vars->viewer->clear();
            vars->image=imagedata->cvimage.clone();
            QImage img(vars->image.data,vars->image.cols,vars->image.rows,vars->image.step,QImage::Format_RGB888);
            vars->viewer->addPixmap(img);
            uint i,n=dpmdata->detection.size();
            for(i=0;i<n;i++)
            {
                vars->viewer->addRect(dpmdata->detection[i].x,dpmdata->detection[i].y
                                      ,dpmdata->detection[i].width,dpmdata->detection[i].height);
            }
        }
        return 0;
    }
}
