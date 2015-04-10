#include<valuebase.h>

using namespace RobotSDK;

NODE_VALUE_BASE_TYPE::NODE_VALUE_BASE_TYPE()
{

}

NODE_VALUE_BASE_TYPE::~NODE_VALUE_BASE_TYPE()
{

}

void NODE_VALUE_BASE_TYPE::loadXMLValues(QString configFileName, QString nodeClass, QString nodeName)
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

NODE_PARAMS_BASE_TYPE::NODE_PARAMS_BASE_TYPE()
{

}

NODE_PARAMS_BASE_TYPE::~NODE_PARAMS_BASE_TYPE()
{

}

QString NODE_PARAMS_BASE_TYPE::getNodeClass()
{
    return _nodeclass;
}

QString NODE_PARAMS_BASE_TYPE::getNodeName()
{
    return _nodename;
}

QString NODE_PARAMS_BASE_TYPE::getExName()
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

NODE_VARS_BASE_TYPE::NODE_VARS_BASE_TYPE()
{

}

NODE_VARS_BASE_TYPE::~NODE_VARS_BASE_TYPE()
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
        if(widgetptr!=widget&&widgetptr->parent()==NULL)
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

void NODE_VARS_BASE_TYPE::setInputPortBufferSize(uint portID, uint bufferSize)
{
    QMutexLocker locker(&_inputportlock);
    if(portID<_inputportnum)
    {
        _buffersize[portID]=bufferSize;
    }
}

void NODE_VARS_BASE_TYPE::setInputPortBufferSize(QList<uint> bufferSize)
{
    QMutexLocker locker(&_inputportlock);
    uint i,n=bufferSize.size();
    for(i=0;i<_inputportnum&&i<n;i++)
    {
        _buffersize[i]=bufferSize.at(i);
    }
}

void NODE_VARS_BASE_TYPE::setInputPortObtainDataBehavior(uint portID, ObtainBehavior obtainDataBehavior)
{
    QMutexLocker locker(&_inputportlock);
    if(portID<_inputportnum)
    {
        _obtaindatabehavior[portID]=obtainDataBehavior;
    }
}

void NODE_VARS_BASE_TYPE::setInputPortObtainDataBehavior(QList<ObtainBehavior> obtainDataBehavior)
{
    QMutexLocker locker(&_inputportlock);
    uint i,n=obtainDataBehavior.size();
    for(i=0;i<_inputportnum&&i<n;i++)
    {
        _obtaindatabehavior[i]=obtainDataBehavior.at(i);
    }
}

void NODE_VARS_BASE_TYPE::setInputPortObtainDataSize(uint portID, uint obtainDataSize)
{
    QMutexLocker locker(&_inputportlock);
    if(portID<_inputportnum)
    {
        _obtaindatasize[portID]=obtainDataSize;
    }
}

void NODE_VARS_BASE_TYPE::setInputPortObtainDataSize(QList<uint> obtainDataSize)
{
    QMutexLocker locker(&_inputportlock);
    uint i,n=obtainDataSize.size();
    for(i=0;i<_inputportnum&&i<n;i++)
    {
        _obtaindatasize[i]=obtainDataSize.at(i);
    }
}

void NODE_VARS_BASE_TYPE::setInputPortTriggerFlag(uint portID, bool triggerFlag)
{
    QMutexLocker locker(&_inputportlock);
    if(portID<_inputportnum)
    {
        _triggerflag[portID]=triggerFlag;
    }
}

void NODE_VARS_BASE_TYPE::setInputPortTriggerFlag(QList<bool> triggerFlag)
{
    QMutexLocker locker(&_inputportlock);
    uint i,n=triggerFlag.size();
    for(i=0;i<_inputportnum&&i<n;i++)
    {
        _triggerflag[i]=triggerFlag.at(i);
    }
}

void NODE_VARS_BASE_TYPE::moveTriggerToPoolThread(QObject * node, QThread * poolThread)
{
    QMap< QString, QObject * >::const_iterator triggeriter;
    for(triggeriter=_qobjecttriggermap.begin();triggeriter!=_qobjecttriggermap.end();triggeriter++)
    {
        if(triggeriter.value()->thread()==node->thread())
        {
            if(_qobjecttriggerpoolthreadflagmap[triggeriter.key()])
            {
                triggeriter.value()->moveToThread(poolThread);
            }
            else
            {
                triggeriter.value()->setParent(node);
            }
        }
    }
}

QWidget *NODE_VARS_BASE_TYPE::getWidget() const
{
    return widget;
}

NodeSwitcher * NODE_VARS_BASE_TYPE::getNodeSwitcher() const
{
    return nodeSwitcher;
}

const QObject *NODE_VARS_BASE_TYPE::getNode()
{
    return _node;
}

NODE_DATA_BASE_TYPE::NODE_DATA_BASE_TYPE()
{

}

NODE_DATA_BASE_TYPE::~NODE_DATA_BASE_TYPE()
{

}

void NODE_DATA_BASE_TYPE::setOutputPortFilterFlag(QList<bool> filterFlag)
{
    _filterflag=filterFlag;
}
