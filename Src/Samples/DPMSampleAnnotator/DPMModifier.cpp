#include"DPMModifier.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, DPMReceiver)
PORT_DECL(1, ROSBagLoader)

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->layout->addWidget(vars->tabwidget);
    vars->tabwidget->addTab(vars->dpmmodifier,"TimeStamp");
    vars->layout->addLayout(vars->controllayout);
    vars->controllayout->addWidget(vars->dpmcontroller);
    vars->controllayout->addStretch();
    vars->controllayout->addWidget(vars->autostart);
    vars->controllayout->addWidget(vars->autostop);
    vars->controllayout->addStretch();
    vars->controllayout->addWidget(vars->trigger);
    vars->widget->setLayout(vars->layout);
    vars->widget->setFocusProxy(vars->dpmmodifier);
    vars->setNodeGUIThreadFlag(1);
	return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    QStringList categories=params->categories.split(",",QString::SkipEmptyParts);
    vars->dpmmodifier->setCategories(categories);
    vars->dpmcontroller->setCategories(categories);
    vars->dataflag=0;
    vars->triggertimer->setInterval(vars->automsec);
    SYNC_CLEAR(vars->dpmsync);
    return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->dataflag=0;
    SYNC_CLEAR(vars->dpmsync);
    vars->triggertimer->stop();
    vars->dpmmodifier->clear();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(IS_INTERNAL_TRIGGER)
    {
        auto data=NODE_DATA;
        if(vars->dataflag)
        {
            data->timestamp=vars->timestamp;
            data->rosbagfile=vars->rosbagfile;
            data->frameid=vars->frameid;
            data->image=vars->image;
            data->saveimageflag=1;
            QVector<DPMRect *> rects=vars->dpmmodifier->getRects();
            int i,n=rects.size();
            for(i=0;i<n;i++)
            {
                DPMData dpmdata;
                dpmdata.category=rects[i]->category;
                dpmdata.id=rects[i]->id;
                dpmdata.rect.x=rects[i]->rect().x();
                dpmdata.rect.y=rects[i]->rect().y();
                dpmdata.rect.width=rects[i]->rect().width();
                dpmdata.rect.height=rects[i]->rect().height();
                dpmdata.attributes.clear();
                data->dpmdata.push_back(dpmdata);
            }
        }
        else
        {
            data->setOutputPortFilterFlag(QList<bool>()<<0<<1);
        }
        vars->dataflag=0;
        return 1;
    }
    else
    {
        bool flag=SYNC_START(vars->dpmsync);
        if(flag)
        {
            auto dpmdata=SYNC_DATA(vars->dpmsync,0);
            auto dpmparams=SYNC_PARAMS(vars->dpmsync,0);
            auto imagedata=SYNC_DATA(vars->dpmsync,1);
            auto imageparams=SYNC_PARAMS(vars->dpmsync,1);

            vars->tabwidget->setTabText(0,QString("TimeStamp: %1, FrameID: %2").arg(dpmdata->timestamp.toString("HH:mm:ss:zzz")).arg(imagedata->frameid));
            if(!vars->dataflag||vars->timestamp!=dpmdata->timestamp)
            {
                vars->dpmmodifier->clear();
                vars->timestamp=dpmdata->timestamp;
                vars->rosbagfile=imageparams->bagfilename;
                vars->frameid=imagedata->frameid;
                vars->image=imagedata->image.clone();
                QImage img(vars->image.data,vars->image.cols,vars->image.rows,vars->image.step,QImage::Format_RGB888);
                vars->dpmmodifier->addPixmap(img);
            }
            uint i,n=dpmdata->detection.size();
            for(i=0;i<n;i++)
            {
                vars->dpmmodifier->addRect(dpmparams->category,dpmdata->id[i]
                                           ,dpmdata->detection[i].x,dpmdata->detection[i].y
                                           ,dpmdata->detection[i].width,dpmdata->detection[i].height);
            }
            vars->dataflag=1;
        }
        return 0;
    }
}
