#include "userinput.h"

UserInput::UserInput(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
	: Source(qstrSharedLibrary,"UserInput",qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
	LoadCheckFptr(sharedlibrary,UIWidgetsFptr,UIWidgets,nodetype,nodeclass);
}

QList<QWidget *> UserInput::getUIWidgets()
{
	QList<QWidget *> widgets;
	UIWidgets(paramsptr.get(),varsptr.get(),widgets);
	if(widgets.size()>0)
	{
		this->moveToThread(QApplication::instance()->thread());
	}
	return widgets;
}