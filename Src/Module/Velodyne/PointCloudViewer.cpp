#include"PointCloudViewer.h"
using namespace RobotSDK_Module;

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VelodyneSensor)

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
    vars->velodynelist=glGenLists(1);
    vars->viewer->addDisplayList(vars->velodynelist);
    Eigen::Matrix4d camerapose=Eigen::Matrix4d::Identity();
    camerapose(2,3)=10;
    vars->viewer->setCameraPose(camerapose);
    vars->viewer->update();
    return 1;
}

//If you don't need manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
    NOUNUSEDWARNING;
    auto vars=NODE_VARS;

    vars->viewer->makeCurrent();
    vars->viewer->deleteDisplayList(vars->velodynelist);
    glDeleteLists(vars->velodynelist,1);
    Eigen::Matrix4d camerapose=Eigen::Matrix4d::Identity();
    camerapose(2,3)=10;
    vars->viewer->setCameraPose(camerapose);
    vars->viewer->update();
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
    glEnableClientState(GL_COLOR_ARRAY);

    GLvoid * pointsptr=(void *)(data->pclpoints->points.data());
    glVertexPointer(3,GL_FLOAT,sizeof(pcl::PointXYZI),pointsptr);

    GLvoid * colorsptr=pointsptr+sizeof(pcl::PointXYZ)+sizeof(float);
    glColorPointer(3,GL_FLOAT,sizeof(pcl::PointXYZI),colorsptr);

    glNewList(vars->velodynelist,GL_COMPILE);

    int pointsnum=data->pclpoints->size();
    glDrawArrays(GL_POINTS,0,pointsnum);

    glEndList();

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    vars->viewer->update();

    return 1;
}
