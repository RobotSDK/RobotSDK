#include"LineExtractorViewer.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, LineExtractor)

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->layout->addWidget(vars->viewer);
    vars->widget->setLayout(vars->layout);
    vars->scene=new QGraphicsScene;
    vars->viewer->setScene(vars->scene);
    vars->setNodeGUIThreadFlag(1);
	return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->scene->clear();
    vars->viewer->resetTransform();
    vars->viewer->scale(vars->ratio,vars->ratio);
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->scene->clear();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=PORT_DATA(0,0);
    vars->scene->clear();

    QLineF xaxis(0,-100,0,100);
    QGraphicsLineItem * xaxisline=new QGraphicsLineItem(xaxis);
    xaxisline->setPen(QPen(Qt::blue, 0.1));
    xaxisline->setZValue(0);
    vars->scene->addItem(xaxisline);

    QLineF yaxis(-100,0,100,0);
    QGraphicsLineItem * yaxisline=new QGraphicsLineItem(yaxis);
    yaxisline->setPen(QPen(Qt::blue, 0.1));
    yaxisline->setZValue(0);
    vars->scene->addItem(yaxisline);

    for(int i=0;i<data->lines.size();i++)
    {
        QLineF startline(0,0,-100*sin(data->lines[i].starttheta),-100*cos(data->lines[i].starttheta));
        QGraphicsLineItem * startlineitem=new QGraphicsLineItem(startline);
        startlineitem->setPen(QPen(Qt::green, 0.1));
        startlineitem->setZValue(0);
        vars->scene->addItem(startlineitem);

        QLineF endline(0,0,-100*sin(data->lines[i].endtheta),-100*cos(data->lines[i].endtheta));
        QGraphicsLineItem * endlineitem=new QGraphicsLineItem(endline);
        endlineitem->setPen(QPen(Qt::green, 0.1));
        endlineitem->setZValue(0);
        vars->scene->addItem(endlineitem);

        for(int j=0;j<data->lines[i].points.size();j++)
        {
            QGraphicsEllipseItem * ellipse=new QGraphicsEllipseItem(-data->lines[i].points[j].y(),-data->lines[i].points[j].x(),0.1,0.1);
            ellipse->setPen(QPen(Qt::red, 0.1));
            ellipse->setZValue(2);
            vars->scene->addItem(ellipse);
        }
        for(int j=0;j<data->lines[i].lines.size();j++)
        {
            QLineF tmpline(-data->lines[i].lines[j].y1(),-data->lines[i].lines[j].x1(),-data->lines[i].lines[j].y2(),-data->lines[i].lines[j].x2());
            QGraphicsLineItem * line=new QGraphicsLineItem(tmpline);
            line->setPen(QPen(Qt::black, 0.1, Qt::DotLine));
            line->setZValue(1);
            vars->scene->addItem(line);
        }
    }
    return 0;
}
