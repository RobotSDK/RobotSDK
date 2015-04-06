#include<Core/ModuleDev/valuebase.h>

using namespace RobotSDK;

XMLValueBase::XMLValueBase()
{

}

XMLValueBase::~XMLValueBase()
{

}

void XMLValueBase::loadXMLValues(QString configName, QString nodeType, QString nodeClass, QString nodeName)
{
    int i,n=_xmlloadfunclist.size();
    if(n>0)
    {
        XMLDomInterface xmlloader(configName,nodeType,nodeClass,nodeName);
        for(i=0;i<n;i++)
        {
            _xmlloadfunclist.at(i)(xmlloader,this);
        }
    }
}
