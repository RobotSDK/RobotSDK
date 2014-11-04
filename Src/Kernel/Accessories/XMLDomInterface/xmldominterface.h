#ifndef XMLDOMINTERFACE_H
#define XMLDOMINTERFACE_H

/*! \addtogroup Accessories
	@{
*/

/*! \file xmldominterface.h
	\brief Defines the class XMLDomInterface.
*/

#include<qdom.h>
#include<qstring.h>
#include<qvector.h>
#include<string>
#include<qfile.h>
#include<qtextstream.h>
#include<qdatetime.h>

/*! \def DEFAULTVALUENAME
	\brief Defines the tag for the default value.
*/
#define DEFAULTVALUENAME "Default"

/*! \def CURTIME_PREDEF
	\brief PreDef for current time.
*/
#define CURTIME_PREDEF QString("$(CurTime)")

/*! \def GetParamValue(loader,params,tag)
	\brief Get parameter params->tag's value from loader. 
*/
#define GetParamValue(loader,params,tag) \
	if(!loader.getParamValue(#tag,params->tag)) \
	{loader.setParamDefault(#tag,params->tag);loader.getParamValue(#tag,params->tag);}

/*! \def GetEnumParamValue(loader,params,tag)
	\brief Get parameter params->tag's enumerate value from loader. 
*/
#define GetEnumParamValue(loader,params,tag) \
	if(!loader.getEnumParamValue(#tag,params->tag)) \
	{loader.setParamDefault(#tag,params->tag);loader.getEnumParamValue(#tag,params->tag);}

/*! \def GetUEnumParamValue(loader,params,tag)
	\brief Get parameter params->tag's unsigned enumerate value from loader. 
*/
#define GetUEnumParamValue(loader,params,tag) \
	if(!loader.getUEnumParamValue(#tag,params->tag)) \
	{loader.setParamDefault(#tag,params->tag);loader.getUEnumParamValue(#tag,params->tag);}

/*! \def GetParamValueEx(loader,params,value,tag)
	\brief Get parameter params->value named tag from loader. 
*/
#define GetParamValueEx(loader,params,value,tag) \
	if(!loader.getParamValue(#tag,params->value)) \
	{loader.setParamDefault(#tag,params->value);loader.getParamValue(#tag,params->value);}

/*! \def GetEnumParamValueEx(loader,params,value,tag)
	\brief Get parameter params->enumerate_value named tag from loader. 
*/
#define GetEnumParamValueEx(loader,params,value,tag) \
	if(!loader.getEnumParamValue(#tag,params->value)) \
	{loader.setParamDefault(#tag,params->value);loader.getEnumParamValue(#tag,params->value);}

/*! \def GetUEnumParamValueEx(loader,params,value,tag)
	\brief Get parameter params->unsigned_enumerate_value named tag from loader. 
*/
#define GetUEnumParamValueEx(loader,params,value,tag) \
	if(!loader.getUEnumParamValue(#tag,params->value)) \
	{loader.setParamDefault(#tag,params->value);loader.getUEnumParamValue(#tag,params->value);}

/*! \class XMLDomInterface
	\brief Class XMLDomInterface is an accessory for Node to load parameters from XML.
	\details
	- XMLDomInterface will use mark(Type_Class_Name) of Node to locate parameters in the XML.
*/
class XMLDomInterface
{
public:
	/*! \fn XMLDomInterface(QString configName, QString nodeType, QString nodeClass, QString nodeName)
		\brief Constructor of XMLDomInterface.
		\param [in] configName The name of config file.
		\param [in] nodeType The type-name of the Node.
		\param [in] nodeClass The class-name of the Node.
		\param [in] nodeName The node-name of the Node.
		\details
		- Load XML file.
		- Locate the parameters.
	*/
	XMLDomInterface(QString configName, QString nodeType, QString nodeClass, QString nodeName);
	/*! \fn ~XMLDomInterface()
		\brief Destructor of XMLDomInterface.
		\details
		If parameters are changed, then the modification will be stored in the XML file. 
	*/
	~XMLDomInterface();
protected:
	/*! \var configname
		\brief The name of config file.
	*/
	QString configname;
	/*! \var doc
		\brief The doc of XML.
	*/
	QDomDocument * doc;
	/*! \var root
		\brief Current root in XML.
	*/
	QDomElement root;
	/*! \var editflag
		\brief The flag to indicate the existence of modification.
	*/
	bool editflag;
	/*! \var nullflag
		\brief The flag to indicate the existence of value.
	*/
	bool nullflag;
protected:
	/*! \fn void getDomRoot(QString tagName)
		\brief Get new \ref root with tagName under current \ref root.
		\param [in] tagName The query name of tag 
	*/
	void getDomRoot(QString tagName);
	/*! \fn void cleanChildren(QDomElement & parentNode)
		\brief Clean the children under certain parent node.
		\param [in] parentNode The parent node.
	*/
	void cleanChildren(QDomElement & parentNode);
	/*! \fn void replacePreDef(QString & value)
		\brief Replace Pre-Def in \a value.
		\param [in,out] value Value to be processed.
	*/
	void replacePreDef(QString & value);
public:
	/*! \fn bool isNull()
		\brief Check whether the current \ref root has value.
	*/
	bool isNull();
public:
	/*! \fn bool exist(QString paramName)
		\brief Whether the parameter paramName exist.
		\param [in] paramName The name of the parameter.
	*/
	bool exist(QString paramName);
	/*! \fn void setParamDefault(QString paramName, ValueType value)
		\brief Set parameter's default value.
		\param [in] paramName The name of the parameter.
		\param [in] value The default value of the parameter.
	*/
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
	/*! \fn void appendParamValue(QString paramName, QString valueName, ValueType value)
		\brief Append parameter's value.
		\param [in] paramName The name of the parameter.
		\param [in] valueName The name of the value.
		\param [in] value The value of the parameter.
	*/
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
	/*! \fn void setParamDefault(QString paramName, QByteArray value)
		\brief Set parameter's default value in QByteArray.
		\param [in] paramName The name of the parameter.
		\param [in] value The default value of the parameter.
	*/
	void setParamDefault(QString paramName, QByteArray value);
	/*! \fn void appendParamValue(QString paramName, QString valueName, QByteArray value)
		\brief Append parameter's value in QByteArray.
		\param [in] paramName The name of the parameter.
		\param [in] valueName The name of the value.
		\param [in] value The value of the parameter.
	*/
	void appendParamValue(QString paramName, QString valueName, QByteArray value);
	/*! \fn bool getParamValue(QString paramName, QString & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's QString value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
	*/
	bool getParamValue(QString paramName, QString & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, bool & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's bool value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
	*/
	bool getParamValue(QString paramName, bool & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, int & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's int value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
		\details
		For base 16, it is recommend to use "unsigned" version. \n
		Support:
		- base 10 : normal number
		- base 16 : "0x" prefixed number
		- base 8 : "0" prefixed number
	*/
	bool getParamValue(QString paramName, int & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, unsigned int & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's unsigned int value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
		\details
		For base 16, it is recommend to use "unsigned" version. \n
		Support:
		- base 10 : normal number
		- base 16 : "0x" prefixed number
		- base 8 : "0" prefixed number
	*/
	bool getParamValue(QString paramName, unsigned int & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, short & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's short value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
		\details
		For base 16, it is recommend to use "unsigned" version. \n
		Support:
		- base 10 : normal number
		- base 16 : "0x" prefixed number
		- base 8 : "0" prefixed number
	*/
	bool getParamValue(QString paramName, short & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, unsigned short & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's unsigned short value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
		\details
		For base 16, it is recommend to use "unsigned" version. \n
		Support:
		- base 10 : normal number
		- base 16 : "0x" prefixed number
		- base 8 : "0" prefixed number
	*/
	bool getParamValue(QString paramName, unsigned short & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, long & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's long value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
		\details
		For base 16, it is recommend to use "unsigned" version. \n
		Support:
		- base 10 : normal number
		- base 16 : "0x" prefixed number
		- base 8 : "0" prefixed number
	*/
	bool getParamValue(QString paramName, long & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, unsigned long & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's unsigned long value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
		\details
		For base 16, it is recommend to use "unsigned" version. \n
		Support:
		- base 10 : normal number
		- base 16 : "0x" prefixed number
		- base 8 : "0" prefixed number
	*/
	bool getParamValue(QString paramName, unsigned long & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, float & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's float value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
	*/
	bool getParamValue(QString paramName, float & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, double & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's double value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
	*/
	bool getParamValue(QString paramName, double & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getParamValue(QString paramName, std::string & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's std::string value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
	*/
	bool getParamValue(QString paramName, std::string & param, QString valueName=QString(DEFAULTVALUENAME));	
	/*! \fn bool getParamValue(QString paramName, QByteArray & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's QByteArray value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
	*/
	bool getParamValue(QString paramName, QByteArray & param, QString valueName=QString(DEFAULTVALUENAME));
	/*! \fn bool getEnumParamValue(QString paramName, EnumType & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's EnumType value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
		\details
		For base 16, it is recommend to use "unsigned" version. \n
		Support:
		- base 10 : normal number
		- base 16 : "0x" prefixed number
		- base 8 : "0" prefixed number
	*/
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
	/*! \fn bool getUEnumParamValue(QString paramName, EnumType & param, QString valueName=QString(DEFAULTVALUENAME))
		\brief Get parameter's EnumType unsigned value.
		\param [in] paramName The name of the parameter.
		\param [out] param The variable to store parameter value.
		\param [in] valueName The name of the value.
		\details
		For base 16, it is recommend to use "unsigned" version. \n
		Support:
		- base 10 : normal number
		- base 16 : "0x" prefixed number
		- base 8 : "0" prefixed number
	*/
	template<class EnumType>
	bool getUEnumParamValue(QString paramName, EnumType & param, QString valueName=QString(DEFAULTVALUENAME))
	{
		unsigned int tempi;
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
	/*! \fn bool getParamValueNameList(QString paramName, QVector<QString> & valueNames, bool hasDefault=1)
		\brief Get list of parameter's values' name.
		\param [in] paramName The name of the parameter.
		\param [out] valueNames The variable to store value's name list.
		\param [in] hasDefault Include default value name or not.
	*/
	bool getParamValueNameList(QString paramName, QVector<QString> & valueNames, bool hasDefault=1);
	/*! \fn bool getParamValueList(QString paramName, QVector<ValueType> & values, bool hasDefault=1)
		\brief Get list of parameter's values' name.
		\param [in] paramName The name of the parameter.
		\param [out] values The variable to store values.
		\param [in] hasDefault Include default value or not.
	*/
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

/*! @}*/

#endif // XMLDOMINTERFACE_H
