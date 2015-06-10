#include"DPMSampleLoader.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, DPMAnnotator)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    QString rosbagfile=params->rosbagfile;
    QFileInfo info(rosbagfile);
    QString sampledir=QString("%1/%2").arg(info.absolutePath()).arg(info.baseName());
    vars->imagesdir=QString("%1/images").arg(sampledir);
    QString samplefile=QString("%1/%2.csv").arg(sampledir).arg(params->samplefilebasename);
    vars->file.setFileName(samplefile);
    if(!vars->file.open(QIODevice::ReadOnly|QIODevice::Text))
    {
        return 0;
    }
    vars->stream.setDevice(&(vars->file));
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->file.close();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;
    if(vars->stream.atEnd())
    {
        return 0;
    }
    QString sample=vars->stream.readLine();
    QStringList valuelist=sample.split(",",QString::SkipEmptyParts);
    if(valuelist.size()<7)
    {
        qDebug()<<"Sample file broken";
        return 0;
    }
    data->frameid=valuelist[0].toInt();
    QString imagefile=QString("%1/%2.png").arg(vars->imagesdir).arg(data->frameid,vars->imagefilenamewidth,10,QChar('0'));
    data->image=cv::imread(imagefile.toStdString());
    data->category=valuelist[1];
    data->id=valuelist[2].toInt();
    data->rect.x=valuelist[3].toInt();
    data->rect.y=valuelist[4].toInt();
    data->rect.width=valuelist[5].toInt();
    data->rect.height=valuelist[6].toInt();
    if(valuelist.size()>7)
    {
        int i,n=valuelist.size();
        for(i=7;i<n;i++)
        {
            data->attributes+=valuelist[i]+QString(",");
        }
        data->attributes.chop(1);
    }
	return 1;
}
