#include<valuebase.h>

using namespace RobotSDK;

NODE_VALUE_BASE_TYPE::NODE_VALUE_BASE_TYPE()
{

}

NODE_VALUE_BASE_TYPE::~NODE_VALUE_BASE_TYPE()
{

}

void NODE_VALUE_BASE_TYPE::loadXMLValues(QString configFileName, QString nodeFullName)
{
    int i,n=_xmlloadfunclist.size();
    if(n>0)
    {
        QStringList nodefullname=nodeFullName.split(QString("::"),QString::SkipEmptyParts);
        XMLDomInterface xmlloader(configFileName,nodefullname);
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

NodeSwitcher::NodeSwitcher(QWidget *parent)
    : QPushButton(parent)
{
    _node=NULL;
    setAutoFillBackground(1);
    connect(this,SIGNAL(clicked()),this,SLOT(slotSwitchNode()),Qt::DirectConnection);
}

void NodeSwitcher::slotSwitchNode()
{
    QEvent * event=new QEvent(QEvent::Type(SwitchEventType));
    QCoreApplication::postEvent(_node,event);
}

void NodeSwitcher::slotNodeState(bool openFlag, QString nodeFullName)
{
    if(openFlag)
    {
        QPalette palette=this->palette();
        palette.setColor(QPalette::Button, QColor(Qt::green));
        this->setPalette(palette);
    }
    else
    {
        QPalette palette=this->palette();
        palette.setColor(QPalette::Button, QColor(Qt::red));
        this->setPalette(palette);
    }
    setText(QString("%1 %2").arg(openFlag?"Close":"Open").arg(nodeFullName));
}

NODE_VARS_BASE_TYPE::NODE_VARS_BASE_TYPE()
{
    widget->setVisible(0);
}

NODE_VARS_BASE_TYPE::~NODE_VARS_BASE_TYPE()
{
    QMap< QString, QObject * >::const_iterator objectiter;
    for(objectiter=_qobjecttriggermap.begin();objectiter!=_qobjecttriggermap.end();objectiter++)
    {
        QObject * objectptr=objectiter.value();
        if(objectptr->parent()==NULL&&!_qobjecttriggerpoolthreadflagmap[objectiter.key()])
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

void XMLVarsBase::setNodeGUIThreadFlag(bool guiThreadFlag)
{
    _guithreadflag=guiThreadFlag;
}

void XMLVarsBase::setNodeShowWidgetFlag(bool showWidgetFlag)
{
    _showwidgetflag=showWidgetFlag;
}

QWidget *XMLVarsBase::getWidget()
{
    return widget;
}

NodeSwitcher *XMLVarsBase::getNodeSwitcher()
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
