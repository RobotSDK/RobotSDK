#include"VelodyneSensor.h"

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Original node functions

//If you don't need initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    auto vars=NODE_VARS;

    if(vars->velodynesub==NULL)
    {
        return 0;
    }
    return 1;
}

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;

    if(!params->calibfilename.isEmpty())
    {
        cv::FileStorage fs;
        fs.open(params->calibfilename.toStdString(),cv::FileStorage::READ);
        if(fs.isOpened())
        {
            fs["VelodyneExtrinsicMat"]>>vars->extrinsicmat;
            fs.release();
        }
        else
        {
            vars->extrinsicmat=cv::Mat::eye(4,4,CV_64F);
        }
    }
    else
    {
        vars->extrinsicmat=cv::Mat::eye(4,4,CV_64F);
    }

    if(vars->velodynesub==NULL)
    {
        return 0;
    }
    vars->velodynesub->resetTopic(vars->topic,vars->queuesize);
    vars->velodynesub->resetQueryInterval(vars->queryinterval);
    vars->velodynesub->startReceiveSlot();
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    auto vars=NODE_VARS;

    vars->velodynesub->stopReceiveSlot();
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto vars=NODE_VARS;
    auto outputdata=NODE_DATA;

    outputdata->rospoints=vars->velodynesub->getMessage();
    if(outputdata->rospoints==NULL)
    {
        return 0;
    }
    int msec=(outputdata->rospoints->header.stamp.sec)%(24*60*60)*1000+(outputdata->rospoints->header.stamp.nsec)/1000000;
    outputdata->timestamp=QTime::fromMSecsSinceStartOfDay(msec);

    outputdata->pclpoints->header.frame_id=outputdata->rospoints->header.frame_id;
    outputdata->pclpoints->header.seq=outputdata->rospoints->header.seq;
    outputdata->pclpoints->header.stamp=msec;

    outputdata->pclpoints->height=outputdata->rospoints->height;
    outputdata->pclpoints->width=outputdata->rospoints->width;

    int pointnum=outputdata->pclpoints->height*outputdata->pclpoints->width;
    outputdata->pclpoints->resize(pointnum);
    char * data=(char *)(outputdata->rospoints->data.data());
    int i,j;
    for(i=0;i<outputdata->rospoints->height;i++)
    {
        for(j=0;j<outputdata->rospoints->width;j++)
        {
            int id=i*outputdata->rospoints->width+j;
            float * tmpdata=(float *)(data+id*(outputdata->rospoints->point_step));
            outputdata->pclpoints->points[id].x=tmpdata[0];
            outputdata->pclpoints->points[id].y=tmpdata[1];
            outputdata->pclpoints->points[id].z=tmpdata[2];
            outputdata->pclpoints->points[id].data[3]=1.0;
            outputdata->pclpoints->points[id].intensity=tmpdata[4]/255.0f;
            outputdata->pclpoints->points[id].data_c[1]=tmpdata[4]/255.0f;
            outputdata->pclpoints->points[id].data_c[2]=tmpdata[4]/255.0f;
            outputdata->pclpoints->points[id].data_c[3]=tmpdata[4]/255.0f;
        }
    }
    outputdata->extrinsicmat=vars->extrinsicmat.clone();
    return 1;
}
