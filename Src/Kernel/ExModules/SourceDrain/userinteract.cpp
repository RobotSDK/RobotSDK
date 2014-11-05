#include "userinteract.h"

UserInteractMono::UserInteractMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
    : SourceDrainMono(qstrSharedLibrary,"UserInteractMono",qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
    LoadCheckFptr(sharedlibrary,UIWidgetsFptr,UIWidgets,nodetype,nodeclass);
}

QList<QWidget *> UserInteractMono::getUIWidgets()
{
    QList<QWidget *> widgets;
    UIWidgets(paramsptr.get(),varsptr.get(),widgets);
    if(widgets.size()>0)
    {
        this->moveToThread(QApplication::instance()->thread());
    }
    return widgets;
}

UserInteractMulti::UserInteractMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx)
    : SourceDrainMulti(qstrSharedLibrary,"UserInteractMulti",qstrNodeClass,qstrNodeName,qstrConfigName,qstrFuncEx)
{
    LoadCheckFptr(sharedlibrary,UIWidgetsFptr,UIWidgets,nodetype,nodeclass);
}

QList<QWidget *> UserInteractMulti::getUIWidgets()
{
    QList<QWidget *> widgets;
    UIWidgets(paramsptr.get(),varsptr.get(),widgets);
    if(widgets.size()>0)
    {
        this->moveToThread(QApplication::instance()->thread());
    }
    return widgets;
}
