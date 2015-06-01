#include "xmldominterface.h"

using namespace RobotSDK;

XMLDomInterface::XMLDomInterface(QString configName, QStringList nodeFullName)
{
    editflag=0;
    nullflag=0;

    QFileInfo fileinfo(configName);
    if(fileinfo.isAbsolute())
    {
        configname=configName;
    }
    else
    {
        QStringList arguments=QApplication::instance()->arguments();
        QString RobotName;
        if(arguments.size()>1)
        {
            RobotName=arguments[1];
        }
        else
        {
            RobotName=QFileInfo(arguments[0]).baseName();
        }
        RobotName.replace(QRegExp("[^a-zA-Z0-9/_$]"),QString("_"));
#ifdef Q_OS_LINUX
        configname=QString("%1/SDK/RobotSDK_%2/Robot-X/%3/%4").arg(QString(qgetenv("HOME"))).arg(ROBOTSDKVER).arg(RobotName).arg(configName);
#endif
#ifdef Q_OS_WIN32
        configname=QString("C:/SDK/RobotSDK_%1/Robot-X/%2/%3").arg(ROBOTSDKVER).arg(XMLDomInterface::RobotName).arg(configName);
#endif
    }
    QDir().mkpath(QFileInfo(configname).path());
    QFile file(configname);
    if(!file.exists())
    {
        doc=new QDomDocument(QString("Configuration"));
        doc->appendChild(doc->createElement("Configuration"));
        uint i,n=nodeFullName.size();
        for(i=0;i<n;i++)
        {
            getDomRoot(nodeFullName.at(i));
        }
    }
    else
    {
        if(file.open(QIODevice::ReadOnly | QIODevice::Text))
        {
            doc=new QDomDocument("Configuration");
            if(!doc->setContent(&file))
            {
                delete doc;
                doc=NULL;
            }
            else
            {
                uint n=nodeFullName.size();
                if(n==2)
                {
                    getDomRoot(nodeFullName.at(0));
                    getDomRoot(nodeFullName.at(1));
                }
                else if(n==3)
                {
                    getDomRoot(QString("%1::%2").arg(nodeFullName.at(0)).arg(nodeFullName.at(2)));
                    getDomRoot(nodeFullName.at(1));
                }
                else
                {
                    doc=NULL;
                }
            }
            file.close();
        }
        else
        {
            doc=NULL;
        }
    }
}

XMLDomInterface::~XMLDomInterface()
{
    if(editflag)
    {
        QFile file(configname);
        if(file.open(QIODevice::WriteOnly|QIODevice::Text))
        {
            QTextStream textstream;
            textstream.setDevice(&file);
            doc->save(textstream,2);
            file.close();
        }
    }
    if(doc!=NULL)
    {
        delete doc;
        doc=NULL;
    }
}

void XMLDomInterface::getDomRoot(QString tagName)
{
    if(doc!=NULL)
    {
        if(root.isNull())
        {
            root=doc->documentElement();
            if(root.isNull())
            {
                editflag=1;
                root=doc->appendChild(doc->createElement(QString("Configuration"))).toElement();
            }
        }
        QDomElement tmpelem=root.firstChildElement(tagName);
        if(tmpelem.isNull())
        {
            nullflag=1;
            editflag=1;
            tmpelem=root.appendChild(doc->createElement(tagName)).toElement();
        }
        else
        {
            nullflag=0;
        }
        root=tmpelem;
    }
}

void XMLDomInterface::cleanChildren(QDomElement & parentNode)
{
    QDomNodeList childlist=parentNode.childNodes();
    int i,n=childlist.size();
    for(i=0;i<n;i++)
    {
        editflag=1;
        parentNode.removeChild(childlist.at(i)).clear();
    }
}

void XMLDomInterface::replacePreDef(QString & value)
{
    QDateTime datetime=QDateTime::currentDateTime();
    QString curtime=datetime.toString(QString("yyyyMMdd_hhmmss_zzz"));
    value.replace(CURTIME_PREDEF,curtime);
    QStringList macrolist;
    if(value.contains("#("))
    {
        int macroindex=value.indexOf("#(");
        while(macroindex>=0)
        {
            int encloseindex=value.indexOf(")",macroindex);
            if(encloseindex>macroindex+2)
            {
                macrolist.push_back(value.mid(macroindex+2,encloseindex-macroindex-2));
                macroindex=value.indexOf("#(",encloseindex);
            }
            else
            {
                break;
            }
        }
        QDomElement macroroot=doc->documentElement().firstChildElement("Macro");
        if(macroroot.isNull())
        {
            macroroot=doc->documentElement().insertBefore(doc->createElement("Macro"),doc->documentElement().firstChildElement()).toElement();
        }
        int i,n=macrolist.size();
        for(i=0;i<n;i++)
        {
            QDomElement macroelement=macroroot.firstChildElement(macrolist.at(i));
            if(macroelement.isNull())
            {
                editflag=1;
                macroelement=macroroot.appendChild(doc->createElement(macrolist.at(i))).toElement();
            }
            if(!macroelement.hasChildNodes())
            {
                macroelement.appendChild(doc->createTextNode(QString("")));
            }
            value.replace(QString("#(%1)").arg(macrolist.at(i)),macroelement.text());
        }
    }
}

bool XMLDomInterface::isNull()
{
    return nullflag;
}

QMap<QString, QList<QString> > XMLDomInterface::getAllParamValues()
{
    QMap<QString, QList<QString> >  result;
    if(!isNull())
    {
        QDomElement paramnode=root.firstChildElement();
        while(!paramnode.isNull())
        {
            if(!result.contains(paramnode.nodeName()))
            {
                result.insert(paramnode.nodeName(),QList<QString>());
            }
            QDomElement valuenode=paramnode.firstChildElement(DEFAULTVALUENAME);
            while(!valuenode.isNull())
            {
                result[paramnode.nodeName()].push_back(valuenode.text());
                valuenode=valuenode.nextSiblingElement();
            }
            paramnode=paramnode.nextSiblingElement();
        }
    }
    return result;
}

void XMLDomInterface::setAllParamValues(QMap<QString, QString> paramValues)
{
    QMap< QString, QString>::const_iterator paramvalueiter;
    for(paramvalueiter=paramValues.begin();paramvalueiter!=paramValues.end();paramvalueiter++)
    {
        setParamDefault(paramvalueiter.key(),paramvalueiter.value());
    }
    editflag=1;
}

bool XMLDomInterface::exist(QString paramName)
{
    if(root.isNull())
    {
        return 0;
    }
    QDomElement paramnode=root.firstChildElement(paramName);
    if(paramnode.isNull())
    {
        return 0;
    }
    return 1;
}

void XMLDomInterface::setParamDefault(QString paramName, QByteArray value)
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
    QString tmpstr(value.toHex());
    valuenode.appendChild(doc->createTextNode(tmpstr));
}

void XMLDomInterface::appendParamValue(QString paramName, QString valueName, QByteArray value)
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
    QString tmpstr(value.toHex());
    valuenode.appendChild(doc->createTextNode(tmpstr));
}

bool XMLDomInterface::getParamValue(QString paramName, QString & param, QString valueName)
{
    if(root.isNull())
    {
        return 0;
    }
    QDomElement paramnode=root.firstChildElement(paramName);
    if(paramnode.isNull())
    {
        return 0;
    }
    QDomElement valuenode=paramnode.firstChildElement(valueName);
    if(valuenode.isNull())
    {
        return 0;
    }
    param=valuenode.text();
    replacePreDef(param);
    return 1;
}

bool XMLDomInterface::getParamValue(QString paramName, bool & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        bool flag;
        int tmp=valuestr.toInt(&flag,0);
        if(flag)
        {
            param=(tmp!=0);
        }
        return flag;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, int & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        bool flag;
        int tmp=valuestr.toInt(&flag,0);
        if(flag)
        {
            param=tmp;
        }
        return flag;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, uint & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        bool flag;
        uint tmp=valuestr.toUInt(&flag,0);
        if(flag)
        {
            param=tmp;
        }
        return flag;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, short & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        bool flag;
        short tmp=valuestr.toShort(&flag,0);
        if(flag)
        {
            param=tmp;
        }
        return flag;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, unsigned short & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        bool flag;
        unsigned short tmp=valuestr.toUShort(&flag,0);
        if(flag)
        {
            param=tmp;
        }
        return flag;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, long & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        bool flag;
        long tmp=valuestr.toLong(&flag,0);
        if(flag)
        {
            param=tmp;
        }
        return flag;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, unsigned long & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        bool flag;
        unsigned long tmp=valuestr.toULong(&flag,0);
        if(flag)
        {
            param=tmp;
        }
        return flag;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, float & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        bool flag;
        float tmp=valuestr.toFloat(&flag);
        if(flag)
        {
            param=tmp;
        }
        return flag;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, double & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        bool flag;
        double tmp=valuestr.toDouble(&flag);
        if(flag)
        {
            param=tmp;
        }
        return flag;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, std::string & param, QString valueName)
{
    QString valuestr;
    if(getParamValue(paramName,valuestr,valueName))
    {
        param=valuestr.toStdString();
        return 1;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValue(QString paramName, QByteArray & param, QString valueName)
{
    QString tmpstr;
    if(getParamValue(paramName,tmpstr,valueName))
    {
        param=QByteArray::fromHex(tmpstr.toUtf8());
        return 1;
    }
    else
    {
        return 0;
    }
}

bool XMLDomInterface::getParamValueNameList(QString paramName, QVector<QString> & valueNames, bool hasDefault)
{
    if(root.isNull())
    {
        return 0;
    }
    QDomElement paramnode=root.firstChildElement(paramName);
    if(paramnode.isNull())
    {
        return 0;
    }
    QDomElement valuenode=paramnode.firstChildElement();
    if(valuenode.isNull())
    {
        return 0;
    }
    while(!valuenode.isNull())
    {
        QString valuename=valuenode.nodeName();
        if(valuename==DEFAULTVALUENAME&&!hasDefault)
        {
            valuenode=valuenode.nextSiblingElement();
            continue;
        }
        valueNames.push_back(valuename);
        valuenode=valuenode.nextSiblingElement();
    }
    return 1;
}
