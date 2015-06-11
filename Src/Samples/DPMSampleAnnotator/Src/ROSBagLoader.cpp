#include"ROSBagLoader.h"
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
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    vars->bag.open(params->bagfilename.toStdString(),rosbag::bagmode::Read);
    vars->view=new rosbag::View(vars->bag,rosbag::TopicQuery(params->bagtopic.toStdString()));
    if(vars->view->size()==0)
    {
        delete vars->view;
        vars->view=NULL;
        return 0;
    }
    vars->viewiter=vars->view->begin();
    vars->curframe=0;
    int count=params->bagstart;
    while(count!=0&&vars->curframe<vars->view->size())
    {
        vars->viewiter++;
        vars->curframe++;
        count--;
    }
    vars->imagepub->resetTopic(vars->rostopic,vars->rosqueuesize);
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->bag.close();
    if(vars->view!=NULL)
    {
        delete vars->view;
        vars->view=NULL;
    }
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    auto data=NODE_DATA;
    sensor_msgs::Image::Ptr image;
    if(vars->curframe<vars->view->size())
    {
        image=(*(vars->viewiter)).instantiate<sensor_msgs::Image>();
    }
    else
    {
        qDebug()<<"Run out of images";
        return 0;
    }
    vars->imagepub->sendMessage(*image);
    int msec=(image->header.stamp.sec)%(24*60*60)*1000+(image->header.stamp.nsec)/1000000;
    data->timestamp=QTime::fromMSecsSinceStartOfDay(msec);
    data->frameid=vars->curframe;
    data->image=cv_bridge::toCvShare(image)->image.clone();
    if(image->encoding=="bgr8")
    {
        cv::cvtColor(data->image,data->image,CV_BGR2RGB);
    }
    int count=params->baginterval;
    while(count!=0&&vars->curframe<vars->view->size())
    {
        vars->viewiter++;
        vars->curframe++;
        count--;
    }
    return 1;
}
