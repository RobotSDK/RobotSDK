#include"PathViewer.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, NDTLocalizer)

//=================================================
//Original node functions

//If you don't need initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->tabwidget->addTab(vars->viewer,"TimeStamp");
    vars->layout->addWidget(vars->tabwidget);
    vars->widget->setLayout(vars->layout);
    vars->setNodeGUIThreadFlag(1);
    return 1;
}

//If you don't need manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->makeCurrent();
    vars->ndtlist=glGenLists(1);
    vars->viewer->addDisplayList(vars->ndtlist);
    Eigen::Matrix4d camerapose=Eigen::Matrix4d::Identity();
    camerapose(2,3)=10;
    vars->viewer->setCameraPose(camerapose);
    vars->viewer->update();
    vars->positions.clear();
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->makeCurrent();
    vars->viewer->deleteDisplayList(vars->ndtlist);
    glDeleteLists(vars->ndtlist,1);
    Eigen::Matrix4d camerapose=Eigen::Matrix4d::Identity();
    camerapose(2,3)=10;
    vars->viewer->setCameraPose(camerapose);
    vars->viewer->update();
    vars->positions.clear();
    return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=PORT_DATA(0,0);

    vars->tabwidget->setTabText(0,data->timestamp.toString("HH:mm:ss:zzz"));

    vars->viewer->makeCurrent();

    glEnableClientState(GL_VERTEX_ARRAY);

    vars->positions.push_back(data->cvtransform.at<double>(0,3));
    vars->positions.push_back(data->cvtransform.at<double>(1,3));
    vars->positions.push_back(data->cvtransform.at<double>(2,3));
    vars->positions.push_back(1);

    int posenum=vars->positions.size()/4;
    cv::Mat pose=cv::Mat(posenum,4,CV_64F,vars->positions.data());
    cv::Mat invpose;
    if(vars->localflag)
    {
        invpose=pose*(data->cvtransform.inv().t());
        glVertexPointer(3,GL_DOUBLE,4*sizeof(double),invpose.data);
    }
    else
    {
        glVertexPointer(3,GL_DOUBLE,4*sizeof(double),pose.data);
    }

    glNewList(vars->ndtlist,GL_COMPILE);

    glColor3f(1.0,1.0,1.0);

    glDrawArrays(GL_LINE_STRIP,0,posenum);

    glEndList();

    glDisableClientState(GL_VERTEX_ARRAY);

    vars->viewer->update();
    return 1;
}
