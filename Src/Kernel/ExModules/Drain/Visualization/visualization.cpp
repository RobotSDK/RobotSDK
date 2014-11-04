#include "visualization.h"

VisualizationMono::VisualizationMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: DrainMono(qstrSharedLibrary,QString("VisualizationMono"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
	LoadCheckFptr(sharedlibrary,visualizationWidgetsFptr,visualizationWidgets,nodetype,nodeclass);
}

QList<QWidget *> VisualizationMono::getVisualizationWidgets()
{
	QList<QWidget *> widgets;
	visualizationWidgets(paramsptr.get(),varsptr.get(),widgets);
	if(widgets.size()>0)
	{
		this->moveToThread(QApplication::instance()->thread());
	}
	return widgets;
}

void VisualizationMono::resetVisualizationSlot()
{
	closeNodeSlot();
	openNodeSlot();
}

VisualizationMulti::VisualizationMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: DrainMulti(qstrSharedLibrary,QString("VisualizationMulti"),qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
	LoadCheckFptr(sharedlibrary,visualizationWidgetsFptr,visualizationWidgets,nodetype,nodeclass);
}

QList<QWidget *> VisualizationMulti::getVisualizationWidgets()
{
	QList<QWidget *> widgets;
	visualizationWidgets(paramsptr.get(),varsptr.get(),widgets);
	if(widgets.size()>0)
	{
		this->moveToThread(QApplication::instance()->thread());
	}
	return widgets;
}

void VisualizationMulti::resetVisualizationSlot()
{
	closeNodeSlot();
	openNodeSlot();
}