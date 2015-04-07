#ifndef VALUEBASE
#define VALUEBASE

#include<Core/ModuleDev/defines.h>

namespace RobotSDK
{

class XMLValueBase
{
    friend class InputPorts;
    friend class QObjectPool;
    friend class OutputPorts;
    friend class Node;
public:
    XMLValueBase();
    virtual ~XMLValueBase();
protected:
    QList< std::function< void(XMLDomInterface &, void *) > > _xmlloadfunclist;
public:
    void loadXMLValues(QString configName, QString nodeType, QString nodeClass, QString nodeName);
};

class XMLParamsBase : public XMLValueBase
{
public:
    XMLParamsBase();
    ~XMLParamsBase();
protected:
    QString _nodetype;
    QString _nodeclass;
    QString _nodename;
public:
    QString getNodeType();
    QString getNodeClass();
    QString getNodeName();
};

class XMLVarsBase : public XMLValueBase
{
public:
    XMLVarsBase();
    ~XMLValueBase();
protected:
    QMutex _inputportlock;
    unsigned int _inputportnum;
    QList< unsigned int > _buffersize;
    QList< ObtainBehavior > _obtaindatabehavior;
    QList< unsigned int > _obtaindatasize;
    QList< bool > _triggerflag;
public:
    void setInputPortBufferSize(unsigned int portID, unsigned int bufferSize);
    void setInputPortBufferSize(QList< unsigned int > bufferSize);
    void setInputPortObtainDataBehavior(unsigned int portID, ObtainBehavior obtainDataBehavior);
    void setInputPortObtainDataBehavior(QList< ObtainBehavior > obtainDataBehavior);
    void setInputPortObtainDataSize(unsigned int portID, unsigned int obtainDataSize);
    void setInputPortObtainDataSize(QList< unsigned int > obtainDataSize);
    void setInputPortTriggerFlag(unsigned int portID, bool triggerFlag);
    void setInputPortTriggerFlag(QList< bool > triggerFlag);
protected:
    unsigned int _outputportnum;
    QList< bool > _filterflag;
public:
    void setOutputPortFilterFlag(unsigned int portID, bool filterFlag);
    void setOutputPortFilterFlag(QList< bool > filterFlag);
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

}

#endif // VALUEBASE

