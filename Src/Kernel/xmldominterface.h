#ifndef XMLDOMINTERFACE_H
#define XMLDOMINTERFACE_H

#include<qdom.h>
#include<QString>
#include<QVector>
#include<QFile>
#include<QTextStream>
#include<QDateTime>
#include<string>
#include<QMap>
#include<QPair>

namespace RobotSDK
{

#define DEFAULTVALUENAME "Default"

#define CURTIME_PREDEF QString("$(CurTime)")

#define GetParamValue(loader,params,tag) \
    if(!loader.getParamValue(#tag,params->tag)) \
{loader.setParamDefault(#tag,params->tag);loader.getParamValue(#tag,params->tag);}

#define GetEnumParamValue(loader,params,tag) \
    if(!loader.getEnumParamValue(#tag,params->tag)) \
{loader.setParamDefault(#tag,params->tag);loader.getEnumParamValue(#tag,params->tag);}

#define GetUEnumParamValue(loader,params,tag) \
    if(!loader.getUEnumParamValue(#tag,params->tag)) \
{loader.setParamDefault(#tag,params->tag);loader.getUEnumParamValue(#tag,params->tag);}

#define GetParamValueEx(loader,params,value,tag) \
    if(!loader.getParamValue(#tag,params->value)) \
{loader.setParamDefault(#tag,params->value);loader.getParamValue(#tag,params->value);}

#define GetEnumParamValueEx(loader,params,value,tag) \
    if(!loader.getEnumParamValue(#tag,params->value)) \
{loader.setParamDefault(#tag,params->value);loader.getEnumParamValue(#tag,params->value);}

#define GetUEnumParamValueEx(loader,params,value,tag) \
    if(!loader.getUEnumParamValue(#tag,params->value)) \
{loader.setParamDefault(#tag,params->value);loader.getUEnumParamValue(#tag,params->value);}

class XMLDomInterface
{
public:
    XMLDomInterface(QString configName, QStringList nodeFullName);
    ~XMLDomInterface();
protected:
    QString configname;
    QDomDocument * doc;
    QDomElement root;
    bool editflag;
    bool nullflag;
protected:
    void getDomRoot(QString tagName);
    void cleanChildren(QDomElement & parentNode);
    void replacePreDef(QString & value);
public:
    bool isNull();
public:
    QMap< QString, QList<QString> > getAllParamValues();
    void setAllParamValues(QMap< QString, QString > paramValues);
public:
    bool exist(QString paramName);
    template<class ValueType>
    void setParamDefault(QString paramName, ValueType value)
    {
        if(root.isNull())
        {
            return;
        }
        QDomElement paramnode=root.firstChildElement(paramName);
        if(paramnode.isNull())
        {
            editflag=1;
            paramnode=root.appendChild(doc->createElement(paramName)).toElement();
        }
        QDomElement valuenode=paramnode.firstChildElement(DEFAULTVALUENAME);
        if(valuenode.isNull())
        {
            editflag=1;
            valuenode=paramnode.appendChild(doc->createElement(DEFAULTVALUENAME)).toElement();
        }
        if(valuenode.hasChildNodes())
        {
            cleanChildren(valuenode);
        }
        editflag=1;
        valuenode.appendChild(doc->createTextNode(QString("%1").arg(value)));
    }
    template<class ValueType>
    void appendParamValue(QString paramName, QString valueName, ValueType value)
    {
        if(root.isNull())
        {
            return;
        }
        QDomElement paramnode=root.firstChildElement(paramName);
        if(paramnode.isNull())
        {
            editflag=1;
            paramnode=root.appendChild(doc->createElement(paramName)).toElement();
        }
        QDomElement valuenode=paramnode.firstChildElement(valueName);
        if(valuenode.isNull())
        {
            editflag=1;
            valuenode=paramnode.appendChild(doc->createElement(valueName)).toElement();
        }
        if(valuenode.hasChildNodes())
        {
            cleanChildren(valuenode);
        }
        editflag=1;
        valuenode.appendChild(doc->createTextNode(QString("%1").arg(value)));
    }
    void setParamDefault(QString paramName, QByteArray value);
    void appendParamValue(QString paramName, QString valueName, QByteArray value);
    bool getParamValue(QString paramName, QString & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, bool & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, int & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, uint & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, short & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, unsigned short & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, long & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, unsigned long & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, float & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, double & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, std::string & param, QString valueName=QString(DEFAULTVALUENAME));
    bool getParamValue(QString paramName, QByteArray & param, QString valueName=QString(DEFAULTVALUENAME));
    template<class EnumType>
    bool getEnumParamValue(QString paramName, EnumType & param, QString valueName=QString(DEFAULTVALUENAME))
    {
        int tempi;
        bool flag=getParamValue(paramName,tempi,valueName);
        if(flag)
        {
            param=EnumType(tempi);
            return 1;
        }
        else
        {
            return 0;
        }
    }
    template<class EnumType>
    bool getUEnumParamValue(QString paramName, EnumType & param, QString valueName=QString(DEFAULTVALUENAME))
    {
        uint tempi;
        bool flag=getParamValue(paramName,tempi,valueName);
        if(flag)
        {
            param=EnumType(tempi);
            return 1;
        }
        else
        {
            return 0;
        }
    }
    bool getParamValueNameList(QString paramName, QVector<QString> & valueNames, bool hasDefault=1);
    template<class ValueType>
    bool getParamValueList(QString paramName, QVector<ValueType> & values, bool hasDefault=1)
    {
        QVector<QString> valuenames;
        if(getParamValueNameList(paramName,valuenames,hasDefault))
        {
            int i,n=valuenames.size();
            for(i=0;i<n;i++)
            {
                ValueType value;
                getParamValue(paramName,value,valuenames[i]);
                values.push_back(value);
            }
        }
        else
        {
            return 0;
        }
    }
};

}

#endif // XMLDOMINTERFACE_H
