#ifndef DEFINES
#define DEFINES

#include<QList>
#include<QVector>
#include<QString>
#include<QMap>
#include<QObject>
#include<QWidget>
#include<QLayout>
#include<QMultiMap>
#include<QPair>
#include<QLibrary>
#include<QtGlobal>
#include<QApplication>
#include<QMutex>
#include<QMutexLocker>
#include<QThread>
#include<QDebug>
#include<QPushButton>
#include<QMetaObject>
#include<QCoreApplication>

#include<deque>
#include<vector>
#include<algorithm>
#include<memory>
#include<functional>

#include<Accessories/XMLDomInterface/xmldominterface.h>

#define RobotSDK_Module
#define RobotSDK_Kernel
//#define RobotSDK_Application

namespace RobotSDK
{

#ifndef RobotSDK_Application

//=================================================================================

enum ObtainBehavior
{
    CopyOldest=0b000,
    GrabOldest=0b001,
    CopyLatest=0b010,
    GrabLatest=0b011,
    CopyOldestStrictly=0b100,
    GrabOldestStrictly=0b101,
    CopyLatestStrictly=0b110,
    GrabLatestStrictly=0b111
};

//=================================================================================

#define XML_VALUE_BASE_TYPE std::shared_ptr< XMLValueBase >
#define XML_VALUE_BASE_CONST_TYPE std::shared_ptr< const XMLValueBase >

#define XML_PARAMS_BASE_TYPE std::shared_ptr< XMLParamsBase >
#define XML_PARAMS_BASE_CONST_TYPE std::shared_ptr< const XMLParamsBase >

#define XML_VARS_BASE_TYPE std::shared_ptr< XMLVarsBase >
#define XML_VARS_BASE_CONST_TYPE std::shared_ptr< const XMLVarsBase >

#define XML_DATA_BASE_TYPE std::shared_ptr< XMLDataBase >
#define XML_DATA_BASE_CONST_TYPE std::shared_ptr< const XMLDataBase >

//=================================================================================

#define TRANSFER_NODE_PARAMS_TYPE XML_PARAMS_BASE_CONST_TYPE
#define TRANSFER_NODE_VARS_TYPE XML_VARS_BASE_TYPE
#define TRANSFER_NODE_DATA_TYPE XML_DATA_BASE_TYPE

#define TRANSFER_PORT_PARAMS_TYPE XML_PARAMS_BASE_CONST_TYPE
#define TRANSFER_PORT_DATA_TYPE XML_DATA_BASE_CONST_TYPE

#define PORT_PARAMS_BUFFER QList< TRANSFER_PORT_PARAMS_TYPE >
#define PORT_DATA_BUFFER QList< TRANSFER_PORT_DATA_TYPE >

#define PORT_PARAMS_CAPSULE QVector< PORT_PARAMS_BUFFER >
#define PORT_DATA_CAPSULE QVector< PORT_DATA_BUFFER >

//=================================================================================

#define INPUT_PARAMS_ARG _inputParams
#define INPUT_DATA_ARG _inputData
#define NODE_PARAMS_ARG _nodeParams
#define NODE_VARS_ARG _nodeVars
#define NODE_DATA_ARG _nodeData

//=================================================================================

#ifdef RobotSDK_Kernel
using namespace RobotSDK;

//=================================================================================
//for Node extendion

#define ROBOTSDK_ARGS_DECL \
    PORT_PARAMS_CAPSULE INPUT_PARAMS_ARG, \
    PORT_DATA_CAPSULE INPUT_DATA_ARG, \
    TRANSFER_NODE_PARAMS_TYPE NODE_PARAMS_ARG, \
    TRANSFER_NODE_VARS_TYPE NODE_VARS_ARG, \
    TRANSFER_NODE_DATA_TYPE NODE_DATA_ARG
#define ROBOTSDK_ARGS \
    INPUT_PARAMS_ARG, \
    INPUT_DATA_ARG, \
    NODE_PARAMS_ARG, \
    NODE_VARS_ARG, \
    NODE_DATA_ARG

#define LOAD_NODE_FUNC_PTR(libraryFileName, nodeClass, funcName) QLibrary::resolve(libraryFileName, QString("%1__%2").arg(nodeClass).arg(#funcName).toUtf8().constData())
#define LOAD_NODE_EXFUNC_PTR(libraryFileName, nodeClass, funcName, exName) QLibrary::resolve(libraryFileName, QString("%1__%2__%3").arg(nodeClass).arg(#funcName).arg(#exName).toUtf8().constData())!=NULL ? \
      QLibrary::resolve(libraryFileName, QString("%1__%2__%3").arg(nodeClass).arg(#funcName).arg(#exName).toUtf8().constData()) \
    : LOAD_NODE_FUNC_PTR(libraryFileName,nodeClass,funcName)

#define ADD_NODE_FUNC_PTR(returnType, funcName, ...) \
    protected: typedef returnType (*funcName##_Fptr)(ROBOTSDK_ARGS_DECL, ##__VA_ARGS__); \
    private: QString _funcptr_##funcName##_Func(){ \
    _funcptrlist.push_back(QString(#funcName)); \
    _funcptrcloadmap.insert(QString(#funcName),[](QString libraryFileName, QString nodeClass, QString exName)->QFunctionPointer{ \
    if(exName.size()==0){return LOAD_NODE_FUNC_PTR(libraryFileName, nodeClass, funcName);} \
    else{return LOAD_NODE_EXFUNC_PTR(libraryFileName, nodeClass, funcName, exName);}}); return QString(#funcName);}; \
    protected: QString funcName=_funcptr_##funcName##_Func();

#define ADD_NODE_DEFAULT_FUNC_PTR(returnType, funcName, ...) ADD_NODE_FUNC_PTR(returnType, funcName, ##__VA_ARGS__)

#define NODE_FUNC_PTR(funcName, ...) (funcName##_Fptr(_funcptrmap[funcName]))(ROBOTSDK_ARGS, ##__VA_ARGS__)

//=================================================================================

#endif

//=================================================================================

#ifdef RobotSDK_Module
#define RobotSDK_EXPORT Q_DECL_EXPORT
#else
#define RobotSDK_EXPORT Q_DECL_IMPORT
#endif

//=================================================================================

#ifdef RobotSDK_Module
using namespace RobotSDK;

//=================================================================================

#define _PARAMS_TYPE ParamsType
#define _VARS_TYPE VarsType
#define _DATA_TYPE DataType

//=================================================================================
//for Node access

#define NODE_PARAMS_TYPE _NODE_PARAMS_TYPE_1(NODE_CLASS)
#define _NODE_PARAMS_TYPE_1(NODE_CLASS) _NODE_PARAMS_TYPE_2(NODE_CLASS)
#define _NODE_PARAMS_TYPE_2(NODE_CLASS) NODE_CLASS##_##_PARAMS_TYPE
#define NODE_PARAMS NODE_PARAMS_ARG ? std::static_pointer_cast< const NODE_PARAMS_TYPE >(NODE_PARAMS_ARG) : std::shared_ptr< const NODE_PARAMS_TYPE >()

#define NODE_VARS_TYPE _NODE_VARS_TYPE_1(NODE_CLASS)
#define _NODE_VARS_TYPE_1(NODE_CLASS) _NODE_VARS_TYPE_2(NODE_CLASS)
#define _NODE_VARS_TYPE_2(NODE_CLASS) NODE_CLASS##_##_VARS_TYPE
#define NODE_VARS NODE_VARS_ARG ? std::static_pointer_cast<NODE_VARS_TYPE>(NODE_VARS_ARG) : std::shared_ptr< NODE_VARS_TYPE >()

#define NODE_DATA_TYPE _NODE_DATA_TYPE_1(NODE_CLASS)
#define _NODE_DATA_TYPE_1(NODE_CLASS) _NODE_DATA_TYPE_2(NODE_CLASS)
#define _NODE_DATA_TYPE_2(NODE_CLASS) NODE_CLASS##_##_DATA_TYPE
#define NODE_DATA NODE_DATA_ARG ? std::static_pointer_cast<NODE_DATA_TYPE>(NODE_DATA_ARG) : std::shared_ptr< NODE_DATA_TYPE >()

//=================================================================================
//for Port access
//portID must be a const number not a variable

#define PORT_PARAMS_TYPE(portID) NODE_CLASS##_INPUT_NODE_##portID##_##_PARAMS_TYPE
#define PORT_DATA_TYPE(portID) NODE_CLASS##_INPUT_NODE_##portID##_##_DATA_TYPE
#define PORT_DECL(portID, inputNodeClass) typedef inputNodeClass##_##_PARAMS_TYPE PORT_PARAMS_TYPE(portID); typedef inputNodeClass##_##_DATA_TYPE PORT_DATA_TYPE(portID);

#define PORT_PARAMS_SIZE(portID) []()->uint{return (portID>=0 && portID<INPUT_PORT_NUM) ? INPUT_PARAMS_ARG[portID].size() : 0;}
#define PORT_PARAMS(portID, paramsID) []()->std::shared_ptr< PORT_PARAMS_TYPE(portID) >{return (paramsID>=0 && paramsID<PORT_PARAMS_SIZE(portID) && INPUT_PARAMS_ARG[portID].at(paramsID)) ? \
      std::static_pointer_cast< const PORT_PARAMS_TYPE(portID) >(INPUT_PARAMS_ARG[portID].at(paramsID)) \
    : std::static_pointer_cast< const PORT_PARAMS_TYPE(portID) >();}

#define PORT_DATA_SIZE(portID) []()->uint{return (portID>=0 && portID<INPUT_PORT_NUM) ? INPUT_DATA_ARG[portID].size() : 0;}
#define PORT_DATA(portID, dataID) []()->std::shared_ptr< PORT_DATA_TYPE(portID) >{return (dataID>=0 && dataID<PORT_DATA_SIZE(portID) && INPUT_DATA_ARG[portID].at(dataID)) ? \
      std::static_pointer_cast< const PORT_DATA_TYPE(portID) >(INPUT_DATA_ARG[portID].at(dataID)) \
    : std::shared_ptr< const PORT_DATA_TYPE(portID) >();}

//=================================================================================
//for XMLValueBase

#define ADD_PARAM(valueType, valueName, valueDefault) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, XMLValueBase * params) \
    {if(!(xmlloader.getParamValue(#valueName,(NODE_PARAMS_TYPE*(params))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_PARAMS_TYPE*(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_ENUM_PARAM(valueType, valueName, valueDefault) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, XMLValueBase * params) \
    {if(!(xmlloader.getEnumParamValue(#valueName,(NODE_PARAMS_TYPE*(params))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_PARAMS_TYPE*(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_UENUM_PARAM(valueType, valueName, valueDefault) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, XMLValueBase * params) \
    {if(!(xmlloader.getUEnumParamValue(#valueName,(NODE_PARAMS_TYPE*(params))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_PARAMS_TYPE*(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_VAR(valueType, valueName, valueDefault) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, XMLValueBase * vars) \
    {if(!(xmlloader.getParamValue(#valueName,(NODE_VARS_TYPE*(vars))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_VARS_TYPE*(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_ENUM_VAR(valueType, valueName, valueDefault) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, XMLValueBase * vars) \
    {if(!(xmlloader.getEnumParamValue(#valueName,(NODE_VARS_TYPE*(vars))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_VARS_TYPE*(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_UENUM_VAR(valueType, valueName, valueDefault) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, XMLValueBase * vars) \
    {if(!(xmlloader.getUEnumParamValue(#valueName,(NODE_VARS_TYPE*(vars))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_VARS_TYPE*(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

//=================================================================================
//for XMLVarsBase

#define ADD_INTERNAL_QOBJECT_TRIGGER(triggerType, triggerName) \
    private: triggerType * _qobject_##triggerType##_##triggerName##_Func() \
    {triggerType * trigger=new triggerType; _qobjecttriggermap.insert(#triggerName, trigger); return trigger;}; \
    public: triggerType * const triggerName=_qobject_##triggerType##_##triggerName##_Func();

#define ADD_INTERNAL_QWIDGET_TRIGGER(triggerType, triggerName) \
    private: triggerType * _qwidget_##triggerType##_##triggerName##_Func() \
    {triggerType * trigger=new triggerType; trigger->moveToThread(QApplication::instance()->thread()); _qwidgettriggermap.insert(#triggerName, trigger); _qwidgetmap.insert(#triggerName, trigger); return trigger;}; \
    public: triggerType * const triggerName=_qwidget_##triggerType##_##triggerName##_Func();

#define ADD_INTERNAL_DEFAULT_CONNECTION(triggeName,signalName) \
    private: QString _default_connection_##triggerName##_##signalName##_Func() \
    {QString connection=QString(SIGNAL(signalName())); _defaultconnectionmap.insert(triggerName,connection); return connection;}; \
    private: QString _default_connection_##triggerName##_##signalName=_default_connection_##triggerName##_##signalName##_Func();

#define ADD_INTERNAL_USER_CONNECTION(triggerName,signalName,slotName,...) \
    private: QPair< QString, QString > _user_connection_##triggerName##_##signalName##_##slotName##_Func() \
    {QPair< QString, QString > connection=QPair< QString, QString >(QString(SIGNAL(signalName(__VA_ARGS__))),QString(SLOT(slotName(__VA_ARGS__)))); \
    _userconnectionmap.insert(triggerName,connection); return connection;}; \
    private: QPair< QString, QString > _user_connection_##triggerName##_##signalName##_##slotName=_user_connection_##triggerName##_##signalName##_##slotName##_Func();

#define ADD_QWIDGET(widgetType, widgetName) \
    private: widgetType * _qwidget_##widgetType##_##widgetName##_Func() \
    {widgetType * widget=new widgetType; widget->moveToThread(QApplication::instance()->thread()); _qwidgetmap.insert(#widgetName, widget); return widget;}; \
    public: widgetType * const widgetName=_qwidget_##widgetType##_##widgetName##_Func();

#define ADD_QLAYOUT(layoutType, layoutName) \
    private: layoutType * _qlayout_##layoutType##_##layoutName##_Func() \
    {layoutType * layout=new layoutType; layout->moveToThread(QApplication::instance()->thread()); _qlayoutmap.insert(#layoutName, layout); return layout;}; \
    public: layoutType * const layoutName=_qlayout_##layoutType##_##layoutName##_Func();

#define ADD_CONNECTION(emitterName,signalName,receiverName,slotName,...) \
    private: QPair< QString, QString > _connection_##emitterName##_##signalName##_##receiverName##_##slotName_Func() \
    {QPair< QString, QString > connection=QPair< QString, QString >(QString(SIGNAL(signalName(__VA_ARGS__))),QString(SLOT(slotName(__VA_ARGS__)))); \
    _connectionmap.insert(QPair< QObject *, QObject * >(emitterName, receiverName), connection); return connection;}; \
    private: QPair< QString, QString > _connection_##emitterName##_##signalName##_##receiverName##_##slotName=_connection_##emitterName##_##signalName##_##receiverName##_##slotName_Func();

//=================================================================================
//for Node Function

#define NODE_FUNC_NAME(funcName) NODE_CLASS##__##funcName
#define NODE_FUNC(funcName, ...) NODE_FUNC_NAME(funcName)(ROBOTSDK_ARGS, ##__VA_ARGS__)
#define NODE_FUNC_DEF(returnType, funcName, ...) returnType NODE_FUNC(funcName, ##__VA_ARGS__)
#define NODE_FUNC_DECL_H(returnType, funcName, ...) extern "C" RobotSDK_EXPORT NODE_FUNC_DEF(returnType, funcName, ##__VA_ARGS__);
#define NODE_FUNC_DECL_CPP(returnType, funcName, ...) extern "C" RobotSDK_EXPORT NODE_FUNC_DEF(returnType, funcName, ##__VA_ARGS__)

#define NODE_EXFUNC_NAME(funcName, exName) NODE_CLASS##__##funcName##__##exName
#define NODE_EXFUNC(funcName, exName, ...) NODE_EXFUNC_NAME(funcName, exName)(ROBOTSDK_ARGS, ##__VA_ARGS__)
#define NODE_EXFUNC_DEF(returnType, funcName, exName, ...) returnType NODE_EXFUNC(funcName, exName, ##__VA_ARGS__)
#define NODE_EXFUNC_DECL_H(returnType, funcName, exName, ...) extern "C" RobotSDK_EXPORT NODE_EXFUNC_DEF(returnType, funcName, exName, ##__VA_ARGS__);
#define NODE_EXFUNC_DECL_CPP(returnType, funcName, exName, ...) extern "C" RobotSDK_EXPORT NODE_EXFUNC_DEF(returnType, funcName, exName, ##__VA_ARGS__)

//=================================================================================
//for default Node Function
#define NODE_DEFAULT_FUNC \
    NODE_FUNC_DECL_CPP(uint, getInputPortNum){ \
    return INPUT_PORT_NUM;} \
    NODE_FUNC_DECL_CPP(uint, getOutputPortNum){ \
    return OUTPUT_POPT_NUM;} \
    NODE_FUNC_DECL_CPP(XML_PARAMS_BASE_TYPE, generateNodeParams){ \
    return XML_PARAMS_BASE_TYPE(new NODE_PARAMS_TYPE);} \
    NODE_FUNC_DECL_CPP(XML_VARS_BASE_TYPE, generateNodeVars){ \
    return XML_VARS_BASE_TYPE(new NODE_VARS_TYPE);} \
    NODE_FUNC_DECL_CPP(XML_DATA_BASE_TYPE, generateNodeData){ \
    return XML_DATA_BASE_TYPE(new NODE_DATA_TYPE);}

//=================================================================================

#endif

#endif

}

#endif // DEFINES

