#ifndef VALUEBASE
#define VALUEBASE

#include<Core/ModuleDev/defines.h>

namespace RobotSDK
{

class XMLValueBase
{
public:
    XMLValueBase();
    virtual ~XMLValueBase();
protected:
    QString _configfilename;
    QList< std::function< void(XMLDomInterface &, XMLValueBase *) > > _xmlloadfunclist;
public:
    void loadXMLValues(QString configName, QString nodeClass, QString nodeName);
};

class XMLParamsBase : public XMLValueBase
{
    friend class Node;
public:
    XMLParamsBase();
    ~XMLParamsBase();
protected:
    QString _nodeclass;
    QString _nodename;
public:
    QString getNodeClass();
    QString getNodeName();
};

class XMLVarsBase : public XMLValueBase
{
    friend class InputPorts;
public:
    XMLVarsBase();
    ~XMLVarsBase();
protected:
    QMutex _inputportlock;
    uint _inputportnum;
    QList< uint > _buffersize;
    QList< ObtainBehavior > _obtaindatabehavior;
    QList< uint > _obtaindatasize;
    QList< bool > _triggerflag;
public:
    void setInputPortBufferSize(uint portID, uint bufferSize);
    void setInputPortBufferSize(QList< uint > bufferSize);
    void setInputPortObtainDataBehavior(uint portID, ObtainBehavior obtainDataBehavior);
    void setInputPortObtainDataBehavior(QList< ObtainBehavior > obtainDataBehavior);
    void setInputPortObtainDataSize(uint portID, uint obtainDataSize);
    void setInputPortObtainDataSize(QList< uint > obtainDataSize);
    void setInputPortTriggerFlag(uint portID, bool triggerFlag);
    void setInputPortTriggerFlag(QList< bool > triggerFlag);
protected:
    QMap< QString, QObject * > _qobjecttriggermap;
    QMap< QString, QObject * > _qwidgettriggermap;
    QMultiMap< QObject *, QString > _defaultconnectionmap;
    QMultiMap< QObject *, QPair< QString, QString > > _userconnectionmap;
    QMultiMap< QPair< QObject *, QObject * > , QPair< QString, QString > > _connectionmap;
    QMap< QString, QWidget * > _qwidgetmap;
    QMap< QString, QLayout * > _qlayoutmap;
public:
    QWidget * mainWidget;
public:
    QObject * getTrigger(QString triggerName);
    QWidget * getWidget(QString widgetName);
    QLayout * getLayout(QString layoutName);
};

class XMLDataBase : public XMLValueBase
{
    friend class OutputPort;
public:
    XMLDataBase();
    ~XMLDataBase();
protected:
    QList< bool > _filterflag;
    uint portid;
public:
    void setOutputPortFilterFlag(QList< bool > filterFlag);
public:
    QTime timestamp;
};

}

#endif // VALUEBASE

