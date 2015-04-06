#ifndef VALUEBASE
#define VALUEBASE

#include<QList>
#include<QVector>
#include<QString>
#include<QReadWriteLock>
#include<QTime>
#include<QMap>
#include<QObject>
#include<QWidget>
#include<QLayout>
#include<QMultiMap>
#include<QCoreApplication>
#include<memory>
#include<functional>
#include<Accessories/XMLDomInterface/xmldominterface.h>
#include<Core/ModuleDev/defines.h>
#include<Core/Port/port.h>

namespace RobotSDK
{

class XMLValueBase
{
    friend class InputPorts;
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
    QMap< QString, QObject * > _objecttriggermap;
    QMap< QString, QObject * > _widgettriggermap;
    QMultiMap< QObject *, QString > _connectionmap;
    QMap< QString, QWidget * > _widgetmap;
    QMap< QString, QLayout * > _layoutmap;
public:
    QWidget * mainWidget;
public:
    QObject * getTrigger(QString triggerName);
    QWidget * getWidget(QString widgetName);
    QLayout * getLayout(QString layoutName);
};

}

#endif // VALUEBASE

