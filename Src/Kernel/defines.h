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
#include<QDockWidget>
#include<QTableWidget>
#include<QTabWidget>
#include<QListWidget>
#include<QEvent>
#include<QFile>

#include<memory>
#include<functional>

#include<xmldominterface.h>

namespace RobotSDK
{

class XMLValueBase;
class XMLParamsBase;
class XMLVarsBase;
class XMLDataBase;

//=================================================================================
//Common Area
//=================================================================================
#define Q_OS_LINUX
#define GLdouble

#ifdef Q_OS_LINUX
#define NUM_0 0b000
#define NUM_1 0b001
#define NUM_2 0b010
#define NUM_3 0b011
#define NUM_4 0b100
#define NUM_5 0b101
#define NUM_6 0b110
#define NUM_7 0b111
#endif

#ifdef Q_OS_WIN
#define NUM_0 0
#define NUM_1 1
#define NUM_2 2
#define NUM_3 3
#define NUM_4 4
#define NUM_5 5
#define NUM_6 6
#define NUM_7 7
#endif

enum ObtainBehavior
{
    CopyOldest=NUM_0,
    GrabOldest=NUM_1,
    CopyLatest=NUM_2,
    GrabLatest=NUM_3,
    CopyOldestStrictly=NUM_4,
    GrabOldestStrictly=NUM_5,
    CopyLatestStrictly=NUM_6,
    GrabLatestStrictly=NUM_7
};

//=================================================================================

#define NODE_VALUE_BASE_TYPE XMLValueBase
#define NODE_PARAMS_BASE_TYPE XMLParamsBase
#define NODE_VARS_BASE_TYPE XMLVarsBase
#define NODE_DATA_BASE_TYPE XMLDataBase

#define XML_VALUE_BASE_TYPE std::shared_ptr< NODE_VALUE_BASE_TYPE >
#define XML_VALUE_BASE_CONST_TYPE std::shared_ptr< const NODE_VALUE_BASE_TYPE >

#define XML_PARAMS_BASE_TYPE std::shared_ptr< NODE_PARAMS_BASE_TYPE >
#define XML_PARAMS_BASE_CONST_TYPE std::shared_ptr< const NODE_PARAMS_BASE_TYPE >

#define XML_VARS_BASE_TYPE std::shared_ptr< NODE_VARS_BASE_TYPE >
#define XML_VARS_BASE_CONST_TYPE std::shared_ptr< const NODE_VARS_BASE_TYPE >

#define XML_DATA_BASE_TYPE std::shared_ptr< NODE_DATA_BASE_TYPE >
#define XML_DATA_BASE_CONST_TYPE std::shared_ptr< const NODE_DATA_BASE_TYPE >

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

#define REGISTER_TRANSFER_VALUE_TYPE(valueType) _REGISTER_TRANSFER_VALUE_TYPE_1(valueType)
#define _REGISTER_TRANSFER_VALUE_TYPE_1(valueType) qRegisterMetaType< valueType >(#valueType);

#define INPUTPORT_SLOT _INPUTPORT_SLOT_1(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE)
#define _INPUTPORT_SLOT_1(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE) _INPUTPORT_SLOT_2(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE)
#define _INPUTPORT_SLOT_2(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE) SLOT(slotReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE))

#define INPUTPORT_SIGNAL _INPUTPORT_SIGNAL_1(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE)
#define _INPUTPORT_SIGNAL_1(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE) _INPUTPORT_SIGNAL_2(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE)
#define _INPUTPORT_SIGNAL_2(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE) SIGNAL(signalReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE, uint))

#define OUTPUTPORT_SLOT _OUTPUTPORT_SLOT_1(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE)
#define _OUTPUTPORT_SLOT_1(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE) _OUTPUTPORT_SLOT_2(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE)
#define _OUTPUTPORT_SLOT_2(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE) SLOT(slotSendParamsData(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE))

#define OUTPUTPORT_SIGNAL _OUTPUTPORT_SIGNAL_1(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE)
#define _OUTPUTPORT_SIGNAL_1(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE) _OUTPUTPORT_SIGNAL_2(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE)
#define _OUTPUTPORT_SIGNAL_2(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE) SIGNAL(signalSendParamsData(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE))

#define INPUTPORTS_SLOT _INPUTPORTS_SLOT_1(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE)
#define _INPUTPORTS_SLOT_1(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE) _INPUTPORTS_SLOT_2(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE)
#define _INPUTPORTS_SLOT_2(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE) SLOT(slotReceiveParamsData(TRANSFER_PORT_PARAMS_TYPE, TRANSFER_PORT_DATA_TYPE, uint))

#define INPUTPORTS_SIGNAL _INPUTPORTS_SIGNAL_1(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE)
#define _INPUTPORTS_SIGNAL_1(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE) _INPUTPORTS_SIGNAL_2(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE)
#define _INPUTPORTS_SIGNAL_2(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE) SIGNAL(signalObtainParamsData(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE))

#define OUTPUTPORTS_SLOT _OUTPUTPORTS_SLOT_1(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE)
#define _OUTPUTPORTS_SLOT_1(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE) _OUTPUTPORTS_SLOT_2(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE)
#define _OUTPUTPORTS_SLOT_2(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE) SLOT(slotSendParamsData(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE))

#define OUTPUTPORTS_SIGNAL _OUTPUTPORTS_SIGNAL_1(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE)
#define _OUTPUTPORTS_SIGNAL_1(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE) _OUTPUTPORTS_SIGNAL_2(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE)
#define _OUTPUTPORTS_SIGNAL_2(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE) SIGNAL(signalSendParamsData(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE))

#define NODE_SLOT _NODE_SLOT_1(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE)
#define _NODE_SLOT_1(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE) _NODE_SLOT_2(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE)
#define _NODE_SLOT_2(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE) SLOT(slotObtainParamsData(PORT_PARAMS_CAPSULE, PORT_DATA_CAPSULE))

#define NODE_SIGNAL _NODE_SIGNAL_1(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE)
#define _NODE_SIGNAL_1(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE) _NODE_SIGNAL_2(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE)
#define _NODE_SIGNAL_2(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE) SIGNAL(signalSendParamsData(TRANSFER_NODE_PARAMS_TYPE, TRANSFER_NODE_DATA_TYPE))

//=================================================================================

#define INPUT_PARAMS_ARG _inputParams
#define INPUT_DATA_ARG _inputData
#define NODE_PARAMS_ARG _nodeParams
#define NODE_VARS_ARG _nodeVars
#define NODE_DATA_ARG _nodeData

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

//=================================================================================

#define SwitchEventType QEvent::User+0
#define OpenNodeEventType QEvent::User+1
#define CloseNodeEventType QEvent::User+2

//=================================================================================

#ifdef RobotSDK_Module
#define RobotSDK_EXPORT Q_DECL_EXPORT
#else
#define RobotSDK_EXPORT Q_DECL_IMPORT
#endif

//=================================================================================
//for NODE_VARS_BASE_TYPE

#define ADD_INTERNAL_QOBJECT_TRIGGER(triggerType, triggerName, poolThreadFlag, ...) \
    private: triggerType * _qobject_##triggerType##_##triggerName##_Func() \
    {triggerType * trigger=new triggerType(__VA_ARGS__); _qobjecttriggermap.insert(#triggerName, trigger); _qobjecttriggerpoolthreadflagmap.insert(#triggerName,poolThreadFlag); return trigger;}; \
    public: triggerType * const triggerName=_qobject_##triggerType##_##triggerName##_Func();

#define ADD_INTERNAL_QWIDGET_TRIGGER(triggerType, triggerName, ...) \
    private: triggerType * _qwidget_##triggerType##_##triggerName##_Func() \
    {triggerType * trigger=new triggerType(__VA_ARGS__); trigger->moveToThread(QApplication::instance()->thread()); _qwidgettriggermap.insert(#triggerName, trigger); _qwidgetmap.insert(#triggerName, trigger); return trigger;}; \
    public: triggerType * const triggerName=_qwidget_##triggerType##_##triggerName##_Func();

#define ADD_INTERNAL_DEFAULT_CONNECTION(triggerName,signalName) \
    private: QString _default_connection_##triggerName##_##signalName##_Func() \
    {QString connection=QString(SIGNAL(signalName())); _defaultconnectionmap.insert(triggerName,connection); return connection;}; \
    private: QString _default_connection_##triggerName##_##signalName=_default_connection_##triggerName##_##signalName##_Func();

#define ADD_INTERNAL_USER_CONNECTION(triggerName,signalName,slotName,...) \
    private: QPair< QString, QString > _user_connection_##triggerName##_##signalName##_##slotName##_Func() \
    {QPair< QString, QString > connection=QPair< QString, QString >(QString(SIGNAL(signalName(__VA_ARGS__))),QString(SLOT(slotName(__VA_ARGS__)))); \
    _userconnectionmap.insert(triggerName,connection); return connection;}; \
    private: QPair< QString, QString > _user_connection_##triggerName##_##signalName##_##slotName=_user_connection_##triggerName##_##signalName##_##slotName##_Func();

#define ADD_QWIDGET(widgetType, widgetName, ...) \
    private: widgetType * _qwidget_##widgetType##_##widgetName##_Func() \
    {widgetType * widget=new widgetType(__VA_ARGS__); widget->moveToThread(QApplication::instance()->thread()); _qwidgetmap.insert(#widgetName, widget); return widget;}; \
    public: widgetType * const widgetName=_qwidget_##widgetType##_##widgetName##_Func();

#define ADD_QLAYOUT(layoutType, layoutName, ...) \
    private: layoutType * _qlayout_##layoutType##_##layoutName##_Func() \
    {layoutType * layout=new layoutType(__VA_ARGS__); layout->moveToThread(QApplication::instance()->thread()); _qlayoutmap.insert(#layoutName, layout); return layout;}; \
    public: layoutType * const layoutName=_qlayout_##layoutType##_##layoutName##_Func();

#define ADD_CONNECTION(emitterName,signalName,receiverName,slotName,...) \
    private: QPair< QString, QString > _connection_##emitterName##_##signalName##_##receiverName##_##slotName_Func() \
    {QPair< QString, QString > connection=QPair< QString, QString >(QString(SIGNAL(signalName(__VA_ARGS__))),QString(SLOT(slotName(__VA_ARGS__)))); \
    _connectionmap.insert(QPair< QObject *, QObject * >(emitterName, receiverName), connection); return connection;}; \
    private: QPair< QString, QString > _connection_##emitterName##_##signalName##_##receiverName##_##slotName=_connection_##emitterName##_##signalName##_##receiverName##_##slotName_Func();

//=================================================================================
//for Node extendion

#define LOAD_NODE_FUNC_PTR(libraryFileName, nodeClass, funcName) QLibrary::resolve(libraryFileName, QString("%1__%2").arg(nodeClass).arg(#funcName).toUtf8().constData())
#define LOAD_NODE_EXFUNC_PTR(libraryFileName, nodeClass, funcName, exName) QLibrary::resolve(libraryFileName, QString("%1__%2__%3").arg(nodeClass).arg(#funcName).arg(exName).toUtf8().constData())!=NULL ? \
      QLibrary::resolve(libraryFileName, QString("%1__%2__%3").arg(nodeClass).arg(#funcName).arg(exName).toUtf8().constData()) \
    : LOAD_NODE_FUNC_PTR(libraryFileName,nodeClass,funcName)

#define ADD_NODE_FUNC_PTR(returnType, funcName, mandatoryFlag, ...) \
    protected: typedef returnType (*funcName##_Fptr)(ROBOTSDK_ARGS_DECL, ##__VA_ARGS__); \
    private: QString _funcptr_##funcName##_Func(){ \
    _funcptrlist.push_back(QString(#funcName)); \
    _funcptrmandatoryflaglist.push_back(mandatoryFlag); \
    _funcptrcloadmap.insert(QString(#funcName),[](QString libraryFileName, QString nodeClass, QString exName)->QFunctionPointer{ \
    if(exName.size()==0){return LOAD_NODE_FUNC_PTR(libraryFileName, nodeClass, funcName);} \
    else{return LOAD_NODE_EXFUNC_PTR(libraryFileName, nodeClass, funcName, exName);}}); return QString(#funcName);}; \
    protected: QString funcName=_funcptr_##funcName##_Func();

#define ADD_NODE_DEFAULT_FUNC_PTR(returnType, funcName, mandatoryFlag, ...) ADD_NODE_FUNC_PTR(returnType, funcName, mandatoryFlag, ##__VA_ARGS__)

#define CHECK_NODE_FUNC_PTR(funcName) _funcptrflag[funcName]
#define NODE_FUNC_PTR(funcName, ...) (funcName##_Fptr(_funcptrmap[funcName]))(ROBOTSDK_ARGS, ##__VA_ARGS__)

//=================================================================================
//Module Area
//=================================================================================
#ifdef RobotSDK_Module

//=================================================================================

#define _PARAMS_TYPE ParamsType
#define _VARS_TYPE VarsType
#define _DATA_TYPE DataType

//=================================================================================
//for Node access

#define NODE_PARAMS_TYPE _NODE_PARAMS_TYPE_1(NODE_CLASS, _PARAMS_TYPE)
#define _NODE_PARAMS_TYPE_1(NODE_CLASS, _PARAMS_TYPE) _NODE_PARAMS_TYPE_2(NODE_CLASS, _PARAMS_TYPE)
#define _NODE_PARAMS_TYPE_2(NODE_CLASS, _PARAMS_TYPE) NODE_CLASS##_##_PARAMS_TYPE
#define NODE_PARAMS NODE_PARAMS_ARG ? std::static_pointer_cast< const NODE_PARAMS_TYPE >(NODE_PARAMS_ARG) : std::shared_ptr< const NODE_PARAMS_TYPE >()

#define NODE_VARS_TYPE _NODE_VARS_TYPE_1(NODE_CLASS, _VARS_TYPE)
#define _NODE_VARS_TYPE_1(NODE_CLASS, _VARS_TYPE) _NODE_VARS_TYPE_2(NODE_CLASS, _VARS_TYPE)
#define _NODE_VARS_TYPE_2(NODE_CLASS, _VARS_TYPE) NODE_CLASS##_##_VARS_TYPE
#define NODE_VARS NODE_VARS_ARG ? std::static_pointer_cast<NODE_VARS_TYPE>(NODE_VARS_ARG) : std::shared_ptr< NODE_VARS_TYPE >()

#define NODE_DATA_TYPE _NODE_DATA_TYPE_1(NODE_CLASS, _DATA_TYPE)
#define _NODE_DATA_TYPE_1(NODE_CLASS, _DATA_TYPE) _NODE_DATA_TYPE_2(NODE_CLASS, _DATA_TYPE)
#define _NODE_DATA_TYPE_2(NODE_CLASS, _DATA_TYPE) NODE_CLASS##_##_DATA_TYPE
#define NODE_DATA NODE_DATA_ARG ? std::static_pointer_cast<NODE_DATA_TYPE>(NODE_DATA_ARG) : std::shared_ptr< NODE_DATA_TYPE >()

#define NODE_PARAMS_TYPE_REF(nodeClass) _NODE_PARAMS_TYPE_REF_1(nodeClass,_PARAMS_TYPE)
#define _NODE_PARAMS_TYPE_REF_1(nodeClass,_PARAMS_TYPE) _NODE_PARAMS_TYPE_REF_2(nodeClass,_PARAMS_TYPE)
#define _NODE_PARAMS_TYPE_REF_2(nodeClass,_PARAMS_TYPE) typedef nodeClass##_##_PARAMS_TYPE NODE_PARAMS_TYPE;

#define NODE_VARS_TYPE_REF(nodeClass) _NODE_VARS_TYPE_REF_1(nodeClass,_VARS_TYPE)
#define _NODE_VARS_TYPE_REF_1(nodeClass,_VARS_TYPE) _NODE_VARS_TYPE_REF_2(nodeClass,_VARS_TYPE)
#define _NODE_VARS_TYPE_REF_2(nodeClass,_VARS_TYPE) typedef nodeClass##_##_VARS_TYPE NODE_VARS_TYPE;

#define NODE_DATA_TYPE_REF(nodeClass) _NODE_DATA_TYPE_REF_1(nodeClass,_DATA_TYPE)
#define _NODE_DATA_TYPE_REF_1(nodeClass,_DATA_TYPE) _NODE_DATA_TYPE_REF_2(nodeClass,_DATA_TYPE)
#define _NODE_DATA_TYPE_REF_2(nodeClass,_DATA_TYPE) typedef nodeClass##_##_DATA_TYPE NODE_DATA_TYPE;

//=================================================================================
//for Port access
//portID must be a const number not a variable

#define PORT_PARAMS_TYPE(portID) _PORT_PARAMS_TYPE_1(portID, _PARAMS_TYPE,NODE_CLASS)
#define _PORT_PARAMS_TYPE_1(portID, _PARAMS_TYPE,NODE_CLASS) _PORT_PARAMS_TYPE_2(portID, _PARAMS_TYPE,NODE_CLASS)
#define _PORT_PARAMS_TYPE_2(portID, _PARAMS_TYPE,NODE_CLASS) NODE_CLASS##_INPUT_NODE_##portID##_##_PARAMS_TYPE

#define PORT_DATA_TYPE(portID) _PORT_DATA_TYPE_1(portID, _DATA_TYPE,NODE_CLASS)
#define _PORT_DATA_TYPE_1(portID, _DATA_TYPE,NODE_CLASS) _PORT_DATA_TYPE_2(portID, _DATA_TYPE,NODE_CLASS)
#define _PORT_DATA_TYPE_2(portID, _DATA_TYPE,NODE_CLASS) NODE_CLASS##_INPUT_NODE_##portID##_##_DATA_TYPE

#define PORT_DECL(portID, inputNodeClass) _PORT_DECL_1(portID, inputNodeClass, _PARAMS_TYPE, _DATA_TYPE)
#define _PORT_DECL_1(portID, inputNodeClass, _PARAMS_TYPE, _DATA_TYPE) _PORT_DECL_2(portID, inputNodeClass, _PARAMS_TYPE, _DATA_TYPE)
#define _PORT_DECL_2(portID, inputNodeClass, _PARAMS_TYPE, _DATA_TYPE) typedef inputNodeClass##_##_PARAMS_TYPE PORT_PARAMS_TYPE(portID); typedef inputNodeClass##_##_DATA_TYPE PORT_DATA_TYPE(portID);

#define PORT_PARAMS_LIST(portID) INPUT_PARAMS_ARG[portID]
#define PORT_PARAMS_SIZE(portID) (portID>=0 && portID<INPUT_PORT_NUM && portID<INPUT_PARAMS_ARG.size()) ? PORT_PARAMS_LIST(portID).size() : 0
#define PORT_PARAMS(portID, paramsID) (paramsID>=0 && paramsID<PORT_PARAMS_SIZE(portID) && PORT_PARAMS_LIST(portID).at(paramsID)) ? \
      std::static_pointer_cast< const PORT_PARAMS_TYPE(portID) >(PORT_PARAMS_LIST(portID).at(paramsID)) \
    : std::shared_ptr< const PORT_PARAMS_TYPE(portID) >()

#define PORT_DATA_LIST(portID) INPUT_DATA_ARG[portID]
#define PORT_DATA_SIZE(portID) (portID>=0 && portID<INPUT_PORT_NUM && portID<INPUT_DATA_ARG.size()) ? PORT_DATA_LIST(portID).size() : 0
#define PORT_DATA(portID, dataID) (dataID>=0 && dataID<PORT_DATA_SIZE(portID) && PORT_DATA_LIST(portID).at(dataID)) ? \
      std::static_pointer_cast< const PORT_DATA_TYPE(portID) >(PORT_DATA_LIST(portID).at(dataID)) \
    : std::shared_ptr< const PORT_DATA_TYPE(portID) >()

#define IS_INTERNAL_TRIGGER INPUT_PARAMS_ARG.size()!=INPUT_PORT_NUM||INPUT_DATA_ARG.size()!=INPUT_PORT_NUM
#define CHECK_VALUE(value) if((value)==NULL){return 0;}

//=================================================================================
//for NODE_VALUE_BASE_TYPE

#define ADD_PARAM(valueType, valueName, valueDefault) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * params) \
    {if(!(xmlloader.getParamValue(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_ENUM_PARAM(valueType, valueName, valueDefault) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * params) \
    {if(!(xmlloader.getEnumParamValue(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_UENUM_PARAM(valueType, valueName, valueDefault) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * params) \
    {if(!(xmlloader.getUEnumParamValue(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_VAR(valueType, valueName, valueDefault) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * vars) \
    {if(!(xmlloader.getParamValue(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_ENUM_VAR(valueType, valueName, valueDefault) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * vars) \
    {if(!(xmlloader.getEnumParamValue(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_UENUM_VAR(valueType, valueName, valueDefault) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * vars) \
    {if(!(xmlloader.getUEnumParamValue(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_OPTIONS(xmlloader, valueName, valueOptions) \
    auto options=valueOptions; uint i,n=options.size(); for(i=0;i<n;i++){ \
    xmlloader.appendParamValue(QString(#valueName),QString("Option_%1").arg(i),options.at(i));}

#define ADD_PARAM_WITH_OPTIONS(valueType, valueName, valueDefault, valueOptions) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * params) \
    {if(!(xmlloader.getParamValue(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName);} \
    ADD_OPTIONS(xmlloader, valueName, valueOptions)});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_ENUM_PARAM_WITH_OPTIONS(valueType, valueName, valueDefault, valueOptions) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * params) \
    {if(!(xmlloader.getEnumParamValue(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName);} \
    ADD_OPTIONS(xmlloader, valueName, valueOptions)});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_UENUM_PARAM_WITH_OPTIONS(valueType, valueName, valueDefault, valueOptions) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * params) \
    {if(!(xmlloader.getUEnumParamValue(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_PARAMS_TYPE *>(params))->valueName);} \
    ADD_OPTIONS(xmlloader, valueName, valueOptions)});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_VAR_WITH_OPTIONS(valueType, valueName, valueDefault, valueOptions) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * vars) \
    {if(!(xmlloader.getParamValue(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName);} \
    ADD_OPTIONS(xmlloader, valueName, valueOptions)});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_ENUM_VAR_WITH_OPTIONS(valueType, valueName, valueDefault, valueOptions) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * vars) \
    {if(!(xmlloader.getEnumParamValue(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName);} \
    ADD_OPTIONS(xmlloader, valueName, valueOptions)});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_UENUM_VAR_WITH_OPTIONS(valueType, valueName, valueDefault, valueOptions) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, NODE_VALUE_BASE_TYPE * vars) \
    {if(!(xmlloader.getUEnumParamValue(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName))) \
    {xmlloader.setParamDefault(QString(#valueName),(static_cast<NODE_VARS_TYPE *>(vars))->valueName);} \
    ADD_OPTIONS(xmlloader, valueName, valueOptions)});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

//=================================================================================
//for Node Function

#define NODE_FUNC_NAME(funcName) _NODE_FUNC_NAME_1(NODE_CLASS,funcName)
#define _NODE_FUNC_NAME_1(NODE_CLASS,funcName) _NODE_FUNC_NAME_2(NODE_CLASS,funcName)
#define _NODE_FUNC_NAME_2(NODE_CLASS,funcName) NODE_CLASS##__##funcName

#define NODE_FUNC(funcName, ...) NODE_FUNC_NAME(funcName)(ROBOTSDK_ARGS, ##__VA_ARGS__)
#define NODE_FUNC_DEF(returnType, funcName, ...) returnType NODE_FUNC_NAME(funcName)(ROBOTSDK_ARGS_DECL, ##__VA_ARGS__)
#define NODE_FUNC_DEF_EXPORT(returnType, funcName, ...) extern "C" RobotSDK_EXPORT NODE_FUNC_DEF(returnType, funcName, ##__VA_ARGS__)
#define NODE_FUNC_DECL(returnType, funcName, ...) extern "C" RobotSDK_EXPORT NODE_FUNC_DEF(returnType, funcName, ##__VA_ARGS__);

#define NODE_EXFUNC_NAME(funcName, exName) _NODE_EXFUNC_NAME_1(NODE_CLASS, funcName, exName)
#define _NODE_EXFUNC_NAME_1(NODE_CLASS, funcName, exName) _NODE_EXFUNC_NAME_2(NODE_CLASS, funcName, exName)
#define _NODE_EXFUNC_NAME_2(NODE_CLASS, funcName, exName) NODE_CLASS##__##funcName##__##exName

#define NODE_EXFUNC(funcName, exName, ...) NODE_EXFUNC_NAME(funcName, exName)(ROBOTSDK_ARGS, ##__VA_ARGS__)
#define NODE_EXFUNC_DEF(returnType, funcName, exName, ...) returnType NODE_EXFUNC_NAME(funcName, exName)(ROBOTSDK_ARGS_DECL, ##__VA_ARGS__)
#define NODE_EXFUNC_DEF_EXPORT(returnType, funcName, exName, ...) extern "C" RobotSDK_EXPORT NODE_EXFUNC_DEF(returnType, funcName, exName, ##__VA_ARGS__)
#define NODE_EXFUNC_DECL(returnType, funcName, exName, ...) extern "C" RobotSDK_EXPORT NODE_EXFUNC_DEF(returnType, funcName, exName, ##__VA_ARGS__);

//=================================================================================
//for default Node Function

#define NODE_DEFAULT_FUNC \
    extern "C" RobotSDK_EXPORT uint NODE_FUNC_NAME(getInputPortNum)(){ \
    return INPUT_PORT_NUM;} \
    extern "C" RobotSDK_EXPORT uint NODE_FUNC_NAME(getOutputPortNum)(){ \
    return OUTPUT_PORT_NUM;} \
    extern "C" RobotSDK_EXPORT XML_PARAMS_BASE_TYPE NODE_FUNC_NAME(generateNodeParams)(){ \
    return XML_PARAMS_BASE_TYPE(new NODE_PARAMS_TYPE);} \
    extern "C" RobotSDK_EXPORT XML_VARS_BASE_TYPE NODE_FUNC_NAME(generateNodeVars)(){ \
    return XML_VARS_BASE_TYPE(new NODE_VARS_TYPE);} \
    extern "C" RobotSDK_EXPORT XML_DATA_BASE_TYPE NODE_FUNC_NAME(generateNodeData)(){ \
    return XML_DATA_BASE_TYPE(new NODE_DATA_TYPE);}

//=================================================================================

#endif

}

#endif // DEFINES

