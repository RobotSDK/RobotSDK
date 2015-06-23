#include"ROSBagLoader_Velodyne.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->layout->addWidget(vars->frame);
    vars->layout->addWidget(vars->next);
    vars->widget->setLayout(vars->layout);
    vars->setNodeGUIThreadFlag(1);
    return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(vars->bagfile_velodyne.size()==0)
    {
        return 0;
    }
    vars->velodynebag.open(vars->bagfile_velodyne.toStdString(),rosbag::bagmode::Read);
    vars->velodyneview=new rosbag::View(vars->velodynebag,rosbag::TopicQuery(vars->bagtopic_velodyne.toStdString()));
    int framenum;
    framenum=vars->velodyneview->size();
    if(framenum==0)
    {
        delete vars->velodyneview;
        vars->velodyneview=NULL;
        return 0;
    }
    if(vars->bagfile_tf==vars->bagfile_velodyne)
    {
        vars->tfflag=1;
        vars->tfview=new rosbag::View(vars->velodynebag,rosbag::TopicQuery(vars->bagtopic_tf.toStdString()));
    }
    else if(vars->bagfile_tf.size()>0)
    {
        vars->tfbag.open(vars->bagfile_tf.toStdString(),rosbag::bagmode::Read);
        vars->tfflag=1;
        vars->tfview=new rosbag::View(vars->tfbag,rosbag::TopicQuery(vars->bagtopic_tf.toStdString()));
    }
    else
    {
        vars->tfflag=0;
        vars->tfview=NULL;
    }
    if(vars->tfflag)
    {
        framenum=vars->tfview->size();
        if(framenum==0)
        {
            delete vars->velodyneview;
            vars->velodyneview=NULL;
            delete vars->tfview;
            vars->tfview=NULL;
            return 0;
        }
    }
    vars->velodyneiter=vars->velodyneview->begin();
    if(vars->tfflag)
    {
        vars->tfiter=vars->tfview->begin();
    }
    vars->curframe=0;
    int count=vars->bagstart;
    while(count>0&&vars->curframe<vars->velodyneview->size())
    {
        vars->velodyneiter++;
        if(vars->tfflag)
        {
            vars->tfiter++;
        }
        count--;
    }
    vars->velodynepub->resetTopic(vars->bagtopic_velodyne,vars->rosqueuesize);
    vars->tfpub->resetFrameID(vars->tf_frameid);
    vars->tfpub->resetChildFrameID(vars->tf_childframeid);

    vars->frame->setText("Frame");
    return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->velodynebag.close();
    if(vars->bagfile_tf!=vars->bagfile_velodyne&&vars->tfflag)
    {
        vars->tfbag.close();
    }
    if(vars->velodyneview!=NULL)
    {
        delete vars->velodyneview;
        vars->velodyneview=NULL;
    }
    if(vars->tfview!=NULL)
    {
        delete vars->tfview;
        vars->tfview=NULL;
    }
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    sensor_msgs::PointCloud2::Ptr velodynemsg;
    if(vars->curframe<vars->velodyneview->size())
    {
        velodynemsg=(*(vars->velodyneiter)).instantiate<sensor_msgs::PointCloud2>();
        vars->velodynepub->sendMessage(*velodynemsg);
        vars->frame->setText(QString("Frame: %1").arg(vars->curframe));
    }
    if(vars->tfflag)
    {
        tf::tfMessage::Ptr tfmsg;
        if(vars->curframe<vars->tfview->size())
        {
            tfmsg=(*(vars->tfiter)).instantiate<tf::tfMessage>();
            int i,n=tfmsg->transforms.size();
            for(i=0;i<n;i++)
            {
                geometry_msgs::TransformStamped msg=tfmsg->transforms[i];
                if(msg.header.frame_id==vars->tf_frameid.toStdString()&&msg.child_frame_id==vars->tf_childframeid.toStdString())
                {
                    tf::StampedTransform tmptf;
                    tf::transformStampedMsgToTF(msg,tmptf);
                    vars->tfpub->sendTF(tmptf);
                    break;
                }
            }
        }
    }
    int count=vars->baginterval;
    while(count>0&&vars->curframe<vars->velodyneview->size())
    {
        vars->velodyneiter++;
        if(vars->tfflag)
        {
            vars->tfiter++;
        }
        vars->curframe++;
        count--;
    }
    return 0;
}
