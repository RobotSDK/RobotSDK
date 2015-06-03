#include"ObstacleMapViewer.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, ObstacleMapGenerator)

//=================================================
//Original node functions

//If you don't need to initialize node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, initializeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->setAlignment(Qt::AlignCenter);
    vars->scrollarea->setWidget(vars->viewer);
    vars->tabwidget->addTab(vars->scrollarea,"TimeStamp");
    vars->layout->addWidget(vars->tabwidget);
    vars->widget->setLayout(vars->layout);

    vars->setNodeGUIThreadFlag(1);
	return 1;
}

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->setText("Open");
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->viewer->setText("Close");
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto data=PORT_DATA(0,0);

    vars->tabwidget->setTabText(0,data->timestamp.toString("HH:mm:ss:zzz"));

    cv::Mat image=data->map;

    QImage img(image.data,image.cols,image.rows,image.step,QImage::Format_RGB888);
    vars->viewer->setPixmap(QPixmap::fromImage(img));
    vars->viewer->resize(img.size());

	return 1;
}
