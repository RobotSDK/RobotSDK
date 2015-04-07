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
#include<memory>
#include<functional>
#include<Accessories/XMLDomInterface/xmldominterface.h>

//#define RobotSDK_ModuleDev

namespace RobotSDK
{

//=================================================================================

enum ObtainBehavior
{
    GrabLatest,
    CopyLatest,
    GrabOldest,
    CopyOldest,
    GrabLatestStrictly,
    CopyLatestStrictly,
    GrabOldestStrictly,
    CopyOldestStrictly
};

//=================================================================================

#define TRANSFER_NODE_PARAM_TYPE std::shared_ptr< const XMLParamsBase >
#define TRANSFER_NODE_VARS_TYPE std::shared_ptr< XMLVarsBase >
#define TRANSFER_TYPE std::shared_ptr< void >
#define TRANSFER_CONST_TYPE std::shared_ptr< const void >
#define PORT_BUFFER QList< TRANSFER_CONST_TYPE >
#define OBTAIN_CAPSULE QVector< PORT_BUFFER >

//=================================================================================

#define ROBOTSDK_ARGS_DECL OBTAIN_CAPSULE _inputParams, OBTAIN_CAPSULE _inputData, TRANSFER_NODE_PARAM_TYPE _nodeParams, TRANSFER_NODE_VARS_TYPE _nodeVars, TRANSFER_TYPE _nodeData
#define ROBOTSDK_ARGS _inputParams, _inputData, _nodeParams, _nodeVars, _nodeData

#define NODE_FUNC_PTR_LOAD(qLibrary, nodeClass, funcName) funcName=(funcName##Fptr)(qLibrary.resolve(QString("%1_%2").arg(nodeClass).arg(#funcName).toUtf8().constData()));
#define NODE_EXFUNC_PTR_LOAD(qLibrary, nodeClass, funcName, exName) funcName=(funcName##Fptr)(qLibrary.resolve(QString("%1_%2_%3").arg(nodeClass).arg(#funcName).arg(#exName).toUtf8().constData())); \
    if(funcName==NULL){NODE_FUNC_PTR_LOAD(qLibrary, nodeClass, funcName)}

#define NODE_FUNC_PTR_ADD(returnType, funcName, ...) \
    protected: typedef returnType (*funcName##Fptr)(ROBOTSDK_ARGS_DECL, __VA_ARGS__); \
    private: funcName##Fptr _funcptr_##funcName##_Func() \
    {_funcptrlist.push_back([](QLibrary & qLibrary, QString nodeClass, QString exName) \
    {if(exName.size()==0){NODE_FUNC_PTR_LOAD(qLibrary, nodeClass, funcName)}else{NODE_EXFUNC_PTR_LOAD(qLibrary, nodeClass, funcName, exName)}}); return NULL;}; \
    protected: funcName##Fptr funcName=_funcptr_##funcName##_Func();

#define NODE_FUNC_PTR_CALL(funcName, ...) funcName(ROBOTSDK_ARGS, __VA_ARGS__)

//=================================================================================

#ifdef RobotSDK_ModuleDev

//=================================================================================

#define _PARAMS_TYPE ParamsType
#define _VARS_TYPE VarsType
#define _DATA_TYPE DataType

//=================================================================================

#define NODE_PARAMS_TYPE _NODE_PARAMS_TYPE_1(NODE_CLASS)
#define _NODE_PARAMS_TYPE_1(NODE_CLASS) _NODE_PARAMS_TYPE_2(NODE_CLASS)
#define _NODE_PARAMS_TYPE_2(NODE_CLASS) NODE_CLASS##_##_PARAMS_TYPE
#define NODE_PARAMS _nodeParams ? std::static_pointer_cast< const NODE_PARAMS_TYPE >(_nodeParams) : std::shared_ptr< const NODE_PARAMS_TYPE >()

#define NODE_VARS_TYPE _NODE_VARS_TYPE_1(NODE_CLASS)
#define _NODE_VARS_TYPE_1(NODE_CLASS) _NODE_VARS_TYPE_2(NODE_CLASS)
#define _NODE_VARS_TYPE_2(NODE_CLASS) NODE_CLASS##_##_VARS_TYPE
#define NODE_VARS _nodeVars ? std::static_pointer_cast<NODE_VARS_TYPE>(_nodeVars) : std::shared_ptr< NODE_VARS_TYPE >()

#define NODE_DATA_TYPE _NODE_DATA_TYPE_1(NODE_CLASS)
#define _NODE_DATA_TYPE_1(NODE_CLASS) _NODE_DATA_TYPE_2(NODE_CLASS)
#define _NODE_DATA_TYPE_2(NODE_CLASS) NODE_CLASS##_##_DATA_TYPE
#define NODE_DATA _nodeData ? std::static_pointer_cast<NODE_DATA_TYPE>(_nodeData) : std::shared_ptr< NODE_DATA_TYPE >()

//=================================================================================

//portID must be a const number not a variable

#define PORT_PARAMS_TYPE(portID) NODE_CLASS##_INPUT_NODE_##portID##_##_PARAMS_TYPE
#define PORT_DATA_TYPE(portID) NODE_CLASS##_INPUT_NODE_##portID##_##_DATA_TYPE
#define PORT_DECL(portID, inputNodeClass) typedef inputNodeClass##_##_PARAMS_TYPE PORT_PARAMS_TYPE(portID); typedef inputNodeClass##_##_DATA_TYPE PORT_DATA_TYPE(portID);

#define PORT_PARAMS_SIZE(portID) []()->unsigned int{return (portID>=0 && portID<INPUT_PORT_NUM) ? _inputParams[portID].size() : 0;}
#define PORT_PARAMS(portID, paramsID) []()->std::shared_ptr< PORT_PARAMS_TYPE(portID) >{return (paramsID>=0 && paramsID<PORT_PARAMS_SIZE(portID) && _inputParams[portID].at(paramsID)) ? \
      std::static_pointer_cast< const PORT_PARAMS_TYPE(portID) >(_inputParams[portID].at(paramsID)) \
    : std::static_pointer_cast< const PORT_PARAMS_TYPE(portID) >();}

#define PORT_DATA_SIZE(portID) []()->unsigned int{return (portID>=0 && portID<INPUT_PORT_NUM) ? _inputData[portID].size() : 0;}
#define PORT_DATA(portID, dataID) []()->std::shared_ptr< PORT_DATA_TYPE(portID) >{return (dataID>=0 && dataID<PORT_DATA_SIZE(portID) && _inputData[portID].at(dataID)) ? \
      std::static_pointer_cast< const PORT_DATA_TYPE(portID) >(_inputData[portID].at(dataID)) \
    : std::shared_ptr< const PORT_DATA_TYPE(portID) >();}

//=================================================================================

#define ADD_PARAM(valueType, valueName, valueDefault) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * params) \
    {if(!(xmlloader.getParamValue(#valueName,(NODE_PARAMS_TYPE*(params))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_PARAMS_TYPE*(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_ENUM_PARAM(valueType, valueName, valueDefault) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * params) \
    {if(!(xmlloader.getEnumParamValue(#valueName,(NODE_PARAMS_TYPE*(params))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_PARAMS_TYPE*(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_UENUM_PARAM(valueType, valueName, valueDefault) \
    private: valueType _params_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * params) \
    {if(!(xmlloader.getUEnumParamValue(#valueName,(NODE_PARAMS_TYPE*(params))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_PARAMS_TYPE*(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_params_##valueType##_##valueName##_Func();

#define ADD_VAR(valueType, valueName, valueDefault) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * vars) \
    {if(!(xmlloader.getParamValue(#valueName,(NODE_VARS_TYPE*(vars))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_VARS_TYPE*(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_ENUM_VAR(valueType, valueName, valueDefault) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * vars) \
    {if(!(xmlloader.getEnumParamValue(#valueName,(NODE_VARS_TYPE*(vars))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_VARS_TYPE*(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_UENUM_VAR(valueType, valueName, valueDefault) \
    private: valueType _vars_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * vars) \
    {if(!(xmlloader.getUEnumParamValue(#valueName,(NODE_VARS_TYPE*(vars))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_VARS_TYPE*(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_vars_##valueType##_##valueName##_Func();

#define ADD_INTERNAL_QOBJECT_TRIGGER(triggerType, triggerName) \
    private: triggerType * _qobject_##triggerType##_##triggerName##_Func() \
    {triggerType * trigger=new triggerType; _qobjecttriggermap.insert(#triggerName, trigger); return trigger;}; \
    public: triggerType * triggerName=_qobject_##triggerType##_##triggerName##_Func();

#define ADD_INTERNAL_QWIDGET_TRIGGER(triggerType, triggerName) \
    private: triggerType * _qwidget_##triggerType##_##triggerName##_Func() \
    {triggerType * trigger=new triggerType; _qwidgettriggermap.insert(#triggerName, trigger); _qwidgetmap.insert(#triggerName, trigger); return trigger;}; \
    public: triggerType * triggerName=_qwidget_##triggerType##_##triggerName##_Func();

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
    {widgetType * widget=new widgetType; _qwidgetmap.insert(#widgetName, widget); return widget;}; \
    public: widgetType * widgetName=_qwidget_##widgetType##_##widgetName##_Func();

#define ADD_QLAYOUT(layoutType, layoutName) \
    private: layoutType * _qlayout_##layoutType##_##layoutName##_Func() \
    {layoutType * layout=new layoutType; _qlayoutmap.insert(#layoutName, layout); return layout;}; \
    public: layoutType * layoutName=_qlayout_##layoutType##_##layoutName##_Func()

#define ADD_CONNECTION(emitterName,signalName,receiverName,slotName,...) \
    private: QPair< QString, QString > _connection_##emitterName##_##signalName##_##receiverName##_##slotName_Func() \
    {QPair< QString, QString > connection=QPair< QString, QString >(QString(SIGNAL(signalName(__VA_ARGS__))),QString(SLOT(slotName(__VA_ARGS__)))); \
    _userconnectionmap.insert(QPair< emitterName, receiverName >, connection); return connection;}; \
    private: QPair< QString, QString > _connection_##emitterName##_##signalName##_##receiverName##_##slotName=_connection_##emitterName##_##signalName##_##receiverName##_##slotName_Func()

//=================================================================================

#define NODE_FUNC_NAME(funcName) NODE_CLASS##__##funcName
#define NODE_FUNC(funcName, ...) NODE_FUNC_NAME(funcName)(ROBOTSDK_ARGS, __VA_ARGS__)
#define NODE_FUNC_DEF(returnType, funcName, ...) returnType NODE_FUNC(funcName, __VA_ARGS__)
#define NODE_FUNC_DECL(returnType, funcName, ...) extern "C" Q_DECL_EXPORT NODE_FUNC_DEF(returnType, funcName, __VA_ARGS__);

#define NODE_EXFUNC_NAME(funcName, exName) NODE_CLASS##__##funcName##_##exName
#define NODE_EXFUNC(funcName, exName, ...) NODE_EXFUNC_NAME(funcName, exName)(ROBOTSDK_ARGS, __VA_ARGS__)
#define NODE_EXFUNC_DEF(returnType, funcName, exName, ...) returnType NODE_EXFUNC(funcName, exName, __VA_ARGS__)
#define NODE_EXFUNC_DECL(returnType, funcName, exName, ...) extern "C" Q_DECL_EXPORT NODE_EXFUNC_DEF(returnType, funcName, exName, __VA_ARGS__);

//=================================================================================

#endif

}

#endif // DEFINES

