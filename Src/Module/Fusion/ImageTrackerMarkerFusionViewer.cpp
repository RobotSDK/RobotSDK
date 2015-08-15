#include"ImageTrackerMarkerFusionViewer.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, ImageTrackerMarkerFusion)

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->layout->addWidget(vars->tab);
    vars->tab->addTab(vars->viewer,"TimeStamp");
    vars->widget->setLayout(vars->layout);
    QGraphicsScene * scene=new QGraphicsScene;
    vars->viewer->setScene(scene);
    vars->setNodeGUIThreadFlag(1);
	return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->scene()->clear();
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->scene()->clear();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=PORT_DATA(0,0);

    vars->tab->setTabText(0,QString("%1 ~ %2").arg(data->timestamp.toString("HH:mm:ss:zzz")).arg(data->trackertimestamp.toString("HH:mm:ss:zzz")));

    vars->viewer->scene()->clear();
    QImage img(data->cvimage.data,data->cvimage.cols,data->cvimage.rows,data->cvimage.step,QImage::Format_RGB888);
    QGraphicsPixmapItem * pixmapitem=vars->viewer->scene()->addPixmap(QPixmap::fromImage(img));
    pixmapitem->setZValue(0);
    int cornernum=data->corners.size();
    for(int i=0;i<cornernum/2;i++)
    {
        QGraphicsLineItem * lineitem=vars->viewer->scene()->addLine(QLineF(data->corners[2*i],data->corners[2*i+1]),QPen(Qt::red,5));
        lineitem->setZValue(1);
    }
	return 1;
}
