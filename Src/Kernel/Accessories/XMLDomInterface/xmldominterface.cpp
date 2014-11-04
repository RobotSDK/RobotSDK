#include "xmldominterface.h"

XMLDomInterface::XMLDomInterface(QString configName, QString nodeType, QString nodeClass, QString nodeName)
{
	editflag=0;
	nullflag=0;

	configname=configName;
	QFile file(configname);
	if(!file.exists())
	{
		doc=new QDomDocument(QString("Configuration"));
		doc->appendChild(doc->createElement("Configuration"));
		getDomRoot(nodeClass);
		getDomRoot(nodeType);
		getDomRoot(nodeName);
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
				getDomRoot(nodeClass);
				getDomRoot(nodeType);				
				getDomRoot(nodeName);
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

bool XMLDomInterface::getParamValue(QString paramName, unsigned int & param, QString valueName)
{
	QString valuestr;
	if(getParamValue(paramName,valuestr,valueName))
	{
		bool flag;
		unsigned int tmp=valuestr.toUInt(&flag,0);
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