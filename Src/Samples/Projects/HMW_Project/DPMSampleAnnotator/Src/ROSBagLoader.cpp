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
    if(params->imageprocflag)
    {
        std::vector<std::string> topics(2);
        topics[0]=params->bagimagetopic.toStdString();
        topics[1]=params->bagcaminfotopic.toStdString();
        vars->view=new rosbag::View(vars->bag,rosbag::TopicQuery(topics));
    }
    else
    {
        vars->view=new rosbag::View(vars->bag,rosbag::TopicQuery(params->bagimagetopic.toStdString()));
    }
    int framenum=vars->view->size();
    if(framenum==0)
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
        if(params->imageprocflag)
        {
            vars->viewiter++;
            vars->viewiter++;
            vars->curframe+=2;
        }
        else
        {
            vars->viewiter++;
            vars->curframe+=1;
        }
        count--;
    }

    vars->imagepub->resetTopic(vars->rosimagetopic,vars->rosqueuesize);
    vars->caminfopub->resetTopic(vars->roscaminfotopic,vars->rosqueuesize);
    vars->imagesub->resetTopic(vars->rosreceiveimagetopic,vars->rosqueuesize);
    vars->imagesub->resetQueryInterval(vars->rosqueryinterval);
    if(params->imageprocflag)
    {
        vars->imagesub->startReceiveSlot();
    }
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto params=NODE_PARAMS;
    auto vars=NODE_VARS;
    vars->bag.close();
    if(vars->view!=NULL)
    {
        delete vars->view;
        vars->view=NULL;
    }
    if(params->imageprocflag)
    {
        vars->imagesub->stopReceiveSlot();
    }
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    if(IS_INTERNAL_TRIGGER)
    {
        auto params=NODE_PARAMS;
        auto vars=NODE_VARS;
        auto data=NODE_DATA;
        sensor_msgs::ImageConstPtr image=vars->imagesub->getMessage();
        qint64 msec=(image->header.stamp.sec)%(24*60*60)*1000+(image->header.stamp.nsec)/1000000;
        data->timestamp=QTime::fromMSecsSinceStartOfDay(msec);
        data->rostimestamp=image->header.stamp.toSec();
        data->frameid=image->header.seq;
        data->image=cv_bridge::toCvShare(image)->image.clone();
        if(params->rgbinvertflag)
        {
            cv::cvtColor(data->image,data->image,CV_BGR2RGB);
        }
        return 1;
    }
    else
    {
        auto params=NODE_PARAMS;
        auto vars=NODE_VARS;
        auto data=NODE_DATA;
        sensor_msgs::Image::Ptr image;
        sensor_msgs::CameraInfo::Ptr caminfo;
        if(vars->curframe<vars->view->size())
        {
            image=(*(vars->viewiter)).instantiate<sensor_msgs::Image>();
            if(image==NULL)
            {
                caminfo=(*(vars->viewiter)).instantiate<sensor_msgs::CameraInfo>();
                vars->caminfopub->sendMessage(*caminfo);
                vars->viewiter++;
                vars->curframe++;
                image=(*(vars->viewiter)).instantiate<sensor_msgs::Image>();
//                if(vars->width>0&&vars->height>0)
//                {
//                    cv_bridge::CvImagePtr tmpimage=cv_bridge::toCvCopy(*image);
//                    cv::Rect roi;
//                    roi.x=vars->x;
//                    roi.y=vars->y;
//                    roi.width=vars->width;
//                    roi.height=vars->height;
//                    tmpimage->image=tmpimage->image(roi);
//                    tmpimage->toImageMsg(*image);
//                }
                vars->imagepub->sendMessage(*image);
                vars->viewiter++;
                vars->curframe++;
            }
            else
            {
//                if(vars->width>0&&vars->height>0)
//                {
//                    cv_bridge::CvImagePtr tmpimage=cv_bridge::toCvCopy(*image);
//                    cv::Rect roi;
//                    roi.x=vars->x;
//                    roi.y=vars->y;
//                    roi.width=vars->width;
//                    roi.height=vars->height;
//                    tmpimage->image=tmpimage->image(roi);
//                    tmpimage->toImageMsg(*image);
//                }
                if(params->imageprocflag)
                {
                    vars->imagepub->sendMessage(*image);
                    vars->viewiter++;
                    vars->curframe++;
                    caminfo=(*(vars->viewiter)).instantiate<sensor_msgs::CameraInfo>();
                    vars->caminfopub->sendMessage(*caminfo);
                    vars->viewiter++;
                    vars->curframe++;
                }
                else
                {
                    vars->imagepub->sendMessage(*image);
                    vars->viewiter++;
                    vars->curframe++;
                }
            }
        }
        else
        {
            qDebug()<<"Run out of images";
            return 0;
        }
        if(!(params->imageprocflag))
        {
            qint64 msec=(image->header.stamp.sec)%(24*60*60)*1000+(image->header.stamp.nsec)/1000000;
            data->timestamp=QTime::fromMSecsSinceStartOfDay(msec);
            data->rostimestamp=image->header.stamp.toSec();
            data->frameid=vars->curframe;
            data->image=cv_bridge::toCvShare(image)->image.clone();
            if(image->encoding=="bgr8")
            {
                cv::cvtColor(data->image,data->image,CV_BGR2RGB);
            }
        }
        int count=params->baginterval-1;
        while(count!=0&&vars->curframe<vars->view->size())
        {
            if(params->imageprocflag)
            {
                vars->viewiter++;
                vars->viewiter++;
                vars->curframe+=2;
            }
            else
            {
                vars->viewiter++;
                vars->curframe+=1;
            }
            count--;
        }
        return !(params->imageprocflag);
    }
}
