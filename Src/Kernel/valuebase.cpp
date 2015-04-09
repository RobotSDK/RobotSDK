#include<valuebase.h>

using namespace RobotSDK;

XMLValueBase::XMLValueBase()
{

}

XMLValueBase::~XMLValueBase()
{

}

void XMLValueBase::loadXMLValues(QString configFileName, QString nodeClass, QString nodeName)
{
    int i,n=_xmlloadfunclist.size();
    if(n>0)
    {
        XMLDomInterface xmlloader(configFileName,nodeClass,nodeName);
        for(i=0;i<n;i++)
        {
            _xmlloadfunclist.at(i)(xmlloader,this);
        }
    }
}

XMLParamsBase::XMLParamsBase()
{

}

XMLParamsBase::~XMLParamsBase()
{

}

QString XMLParamsBase::getNodeClass()
{
    return _nodeclass;
}

QString XMLParamsBase::getNodeName()
{
    return _nodename;
}

QString XMLParamsBase::getExName()
{
    return _exname;
}

const int NodeSwitcher::SwitchEventType=QEvent::registerEventType();
const int NodeSwitcher::OpenNodeEventType=QEvent::registerEventType();
const int NodeSwitcher::CloseNodeEventType=QEvent::registerEventType();

NodeSwitcher::NodeSwitcher(QWidget *parent)
    : QPushButton(parent)
{
    _node=NULL;
    connect(this,SIGNAL(clicked()),this,SLOT(slotSwitchNode()),Qt::DirectConnection);
}

void NodeSwitcher::slotSwitchNode()
{
    QEvent * event=new QEvent(QEvent::Type(SwitchEventType));
    QCoreApplication::postEvent(_node,event);
}

WidgetSwitcher::WidgetSwitcher(QWidget * parent)
    : QPushButton(parent)
{
    visibleflag=0;
    connect(this,SIGNAL(clicked()),this,SLOT(slotSwitchWidget()),Qt::DirectConnection);
}

void WidgetSwitcher::slotSwitchWidget()
{
    visibleflag=!visibleflag;
    emit signalSwitchWidget(visibleflag);
}

XMLVarsBase::XMLVarsBase()
{
    widget->setVisible(0);
}

XMLVarsBase::~XMLVarsBase()
{
    QMap< QString, QObject * >::const_iterator objectiter;
    for(objectiter=_qobjecttriggermap.begin();objectiter!=_qobjecttriggermap.end();objectiter++)
    {
        QObject * objectptr=objectiter.value();
        if(objectptr->parent()==NULL)
        {
            delete objectptr;
        }
    }
    _qobjecttriggermap.clear();
    for(objectiter=_qwidgettriggermap.begin();objectiter!=_qwidgettriggermap.end();objectiter++)
    {
        QObject * objectptr=objectiter.value();
        if(objectptr->parent()==NULL)
        {
            delete objectptr;
        }
    }
    _qobjecttriggermap.clear();

    QMap< QString, QWidget * >::const_iterator widgetiter;
    for(widgetiter=_qwidgetmap.begin();widgetiter!=_qwidgetmap.end();widgetiter++)
    {
        QWidget * widgetptr=widgetiter.value();
        if(widgetptr->parent()==NULL)
        {
            delete widgetptr;
        }
    }
    _qwidgetmap.clear();

    QMap< QString, QLayout * >::const_iterator layoutiter;
    for(layoutiter=_qlayoutmap.begin();layoutiter!=_qlayoutmap.end();layoutiter++)
    {
        QLayout * layoutptr=layoutiter.value();
        if(layoutptr->parent()==NULL)
        {
            delete layoutptr;
        }
    }
    _qlayoutmap.clear();

    if(widget->parent()==NULL)
    {
        delete widget;
    }
}

void XMLVarsBase::setInputPortBufferSize(uint portID, uint bufferSize)
{
    QMutexLocker locker(&_inputportlock);
    if(portID<_inputportnum)
    {
        _buffersize[portID]=bufferSize;
    }
}

void XMLVarsBase::setInputPortBufferSize(QList<uint> bufferSize)
{
    QMutexLocker locker(&_inputportlock);
    uint i,n=bufferSize.size();
    for(i=0;i<_inputportnum&&i<n;i++)
    {
        _buffersize[i]=bufferSize.at(i);
    }
}

void XMLVarsBase::setInputPortObtainDataBehavior(uint portID, ObtainBehavior obtainDataBehavior)
{
    QMutexLocker locker(&_inputportlock);
    if(portID<_inputportnum)
    {
        _obtaindatabehavior[portID]=obtainDataBehavior;
    }
}

void XMLVarsBase::setInputPortObtainDataBehavior(QList<ObtainBehavior> obtainDataBehavior)
{
    QMutexLocker locker(&_inputportlock);
    uint i,n=obtainDataBehavior.size();
    for(i=0;i<_inputportnum&&i<n;i++)
    {
        _obtaindatabehavior[i]=obtainDataBehavior.at(i);
    }
}

void XMLVarsBase::setInputPortObtainDataSize(uint portID, uint obtainDataSize)
{
    QMutexLocker locker(&_inputportlock);
    if(portID<_inputportnum)
    {
        _obtaindatasize[portID]=obtainDataSize;
    }
}

void XMLVarsBase::setInputPortObtainDataSize(QList<uint> obtainDataSize)
{
    QMutexLocker locker(&_inputportlock);
    uint i,n=obtainDataSize.size();
    for(i=0;i<_inputportnum&&i<n;i++)
    {
        _obtaindatasize[i]=obtainDataSize.at(i);
    }
}

void XMLVarsBase::setInputPortTriggerFlag(uint portID, bool triggerFlag)
{
    QMutexLocker locker(&_inputportlock);
    if(portID<_inputportnum)
    {
        _triggerflag[portID]=triggerFlag;
    }
}

void XMLVarsBase::setInputPortTriggerFlag(QList<bool> triggerFlag)
{
    QMutexLocker locker(&_inputportlock);
    uint i,n=triggerFlag.size();
    for(i=0;i<_inputportnum&&i<n;i++)
    {
        _triggerflag[i]=triggerFlag.at(i);
    }
}

void XMLVarsBase::moveTriggerToPoolThread(QObject * node, QThread * poolThread)
{
    QMap< QString, QObject * >::const_iterator triggeriter;
    for(triggeriter=_qobjecttriggermap.begin();triggeriter!=_qobjecttriggermap.end();triggeriter++)
    {
        if(triggeriter.value()->thread()==node->thread())
        {
            triggeriter.value()->moveToThread(poolThread);
        }
    }
}

QWidget *XMLVarsBase::getWidget() const
{
    return widget;
}

NodeSwitcher * XMLVarsBase::getNodeSwitcher() const
{
    return nodeSwitcher;
}

WidgetSwitcher * XMLVarsBase::getWidgetSwitcher() const
{
    return widgetSwitcher;
}

const QObject *XMLVarsBase::getNode()
{
    return _node;
}

XMLDataBase::XMLDataBase()
{

}

XMLDataBase::~XMLDataBase()
{

}

void XMLDataBase::setOutputPortFilterFlag(QList<bool> filterFlag)
{
    _filterflag=filterFlag;
}
