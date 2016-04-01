#include"DPMSampleSaver.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, DPMModifier)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->openfileflag=0;
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(vars->file.isOpen())
    {
        vars->file.close();
    }
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=PORT_DATA(0,0);
    if(!vars->openfileflag)
    {
        QString rosbagfile=data->rosbagfile;
        QFileInfo info(rosbagfile);
        QString sampledir=QString("%1/%2").arg(info.absolutePath()).arg(info.baseName());
        QDir().mkpath(sampledir);
        vars->imagesdir=QString("%1/images").arg(sampledir);
        QDir().mkpath(vars->imagesdir);
        QString samplefile=QString("%1/samples_%2.csv").arg(sampledir).arg(QDateTime::currentDateTime().toString("yyyyMMddHHmmss"));
        vars->file.setFileName(samplefile);
        vars->file.open(QIODevice::WriteOnly|QIODevice::Text);
        vars->openfileflag=1;
    }
    if(data->saveimageflag)
    {
        QString imagefilename=QString("%1/%2.png").arg(vars->imagesdir).arg(data->frameid,vars->imagefilenamewidth,10,QChar('0'));
        cv::imwrite(imagefilename.toStdString(),data->image);
    }
    int i,n=data->dpmdata.size();
    for(i=0;i<n;i++)
    {
        QString sampleinfo=QString("%1,%2,%3,%4,%5,%6,%7,%8")
                .arg(data->rostimestamp).arg(data->frameid).arg(data->dpmdata[i].category).arg(data->dpmdata[i].id)
                .arg(data->dpmdata[i].rect.x).arg(data->dpmdata[i].rect.y)
                .arg(data->dpmdata[i].rect.width).arg(data->dpmdata[i].rect.height);
        if(data->dpmdata[i].attributes.size()>0)
        {
            sampleinfo+=QString(",")+data->dpmdata[i].attributes;
            if(sampleinfo.endsWith(","))
            {
                sampleinfo.chop(1);
            }
        }
        sampleinfo+=QString("\n");
        vars->file.write(sampleinfo.toUtf8());
    }
    vars->file.flush();
    return 0;
}
