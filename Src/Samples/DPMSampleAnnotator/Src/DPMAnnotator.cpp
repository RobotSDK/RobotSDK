#include"DPMAnnotator.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, DPMSampleLoader)

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->layout->addWidget(vars->annotator);
    vars->widget->setLayout(vars->layout);
    vars->setNodeGUIThreadFlag(1);
	return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    vars->annotator->clearAttributes();
    QFile file(vars->attributesfile);
    if(file.exists()&&file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        QTextStream stream(&file);
        while(!stream.atEnd())
        {
            QStringList categories=stream.readLine().split(",",QString::SkipEmptyParts);
            QString attribute=stream.readLine();
            int i,n=categories.size();
            for(i=0;i<n;i++)
            {
                vars->annotator->setAttributes(categories[i],attribute);
            }
        }
        file.close();
        vars->categories=params->categories.split(",",QString::SkipEmptyParts);
        if(vars->categories.size()==0)
        {
            return 0;
        }
        return 1;
    }
    else
    {
        return 0;
    }
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->annotator->clearAttributes();
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
        QString attributes=vars->annotator->getAttributes();
        if(vars->dataflag&&attributes.size()>0)
        {
            data->rosbagfile=vars->rosbagfile;
            data->frameid=vars->frameid;
            data->saveimageflag=0;
            data->dpmdata.resize(1);
            data->dpmdata[0].category=vars->category;
            data->dpmdata[0].id=vars->id;
            data->dpmdata[0].rect=vars->rect;
            data->dpmdata[0].attributes=attributes;
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
        auto params=PORT_PARAMS(0,0);
        auto data=PORT_DATA(0,0);
        if(vars->categories.contains(data->category))
        {
            vars->rosbagfile=params->rosbagfile;
            vars->frameid=data->frameid;
            vars->category=data->category;
            vars->id=data->id;
            vars->rect=data->rect;
            cv::Mat image;
            data->image.convertTo(image,-1,vars->alpha,vars->beta);
            if(vars->rgbconvertflag)
            {
                cv::cvtColor(image,image,CV_BGR2RGB);
            }
            vars->annotator->showSample(image,data->frameid,data->category,data->id,data->rect,data->attributes);
            vars->dataflag=1;
            return 0;
        }
        else
        {
            auto outputdata=NODE_DATA;
            outputdata->setOutputPortFilterFlag(QList<bool>()<<0<<1);
            return 1;
        }
    }
}
