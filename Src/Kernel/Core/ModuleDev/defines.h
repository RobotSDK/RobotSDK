#ifndef DEFINES
#define DEFINES

#include<QList>
#include<QVector>
#include<QString>
#include<QtGlobal>
#include<memory>
#include<functional>
#include<Accessories/XMLDomInterface/xmldominterface.h>

#define RobotSDK_ModuleDev

namespace RobotSDK
{

//=================================================================================

#define TRANSFER_NODE_PARAM_TYPE std::shared_ptr< const XMLParamsBase >
#define TRANSFER_NODE_VARS_TYPE std::shared_ptr< XMLVarsBase >
#define TRANSFER_TYPE std::shared_ptr< void >
#define TRANSFER_CONST_TYPE std::shared_ptr< const void >
#define PORT_BUFFER QList< TRANSFER_CONST_TYPE >
#define OBTAIN_CAPSULE QVector< PORT_BUFFER >

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

#define ROBOTSDK_ARGS_DECL OBTAIN_CAPSULE _inputParams, OBTAIN_CAPSULE _inputData, TRANSFER_NODE_PARAM_TYPE _nodeParams, TRANSFER_NODE_VARS_TYPE _nodeVars, TRANSFER_TYPE _nodeData
#define ROBOTSDK_ARGS _inputParams, _inputData, _nodeParams, _nodeVars, _nodeData

#define NODE_FUNC_PTR_DECL(returnType, funcName, ...) protected: typedef returnType (*funcName##Fptr)(ROBOTSDK_ARGS_DECL, __VA_ARGS__); funcName##Fptr funcName=NULL;
#define NODE_FUNC_PTR_LOAD(funcName, qLibrary, nodeType, nodeClass) funcName=(funcName##Fptr)(qLibrary.resolve(QString("%1_%2_%3").arg(nodeType).arg(nodeClass).arg(#funcName).toUtf8().constData()));
#define NODE_FUNC_PTR_CALL(funcName, ...) funcName(ROBOTSDK_ARGS, __VA_ARGS__)

#define NODE_EXFUNC_PTR_LOAD(funcName, exName, qLibrary, nodeType, nodeClass) funcName=(funcName##Fptr)(qLibrary.resolve(QString("%1_%2_%3_%4").arg(nodeType).arg(nodeClass).arg(#funcName).arg(#exName).toUtf8().constData())); \
    if(funcName==NULL){NODE_FUNC_PTR_LOAD(funcName, qLibrary, nodeType, nodeClass)}

//=================================================================================

#ifdef RobotSDK_ModuleDev

//=================================================================================

#define NODE_PARAMS_TYPE _NODE_PARAMS_TYPE_1(NODE_TYPE, NODE_CLASS)
#define _NODE_PARAMS_TYPE_1(NODE_TYPE, NODE_CLASS) _NODE_PARAMS_TYPE_2(NODE_TYPE, NODE_CLASS)
#define _NODE_PARAMS_TYPE_2(NODE_TYPE, NODE_CLASS) NODE_TYPE##_##NODE_CLASS##_Params
#define NODE_PARAMS _nodeParams ? std::static_pointer_cast< const NODE_PARAMS_TYPE >(_nodeParams) : std::shared_ptr< const NODE_PARAMS_TYPE >()

#define NODE_VARS_TYPE _NODE_VARS_TYPE_1(NODE_TYPE, NODE_CLASS)
#define _NODE_VARS_TYPE_1(NODE_TYPE, NODE_CLASS) _NODE_VARS_TYPE_2(NODE_TYPE, NODE_CLASS)
#define _NODE_VARS_TYPE_2(NODE_TYPE, NODE_CLASS) NODE_TYPE##_##NODE_CLASS##_Vars
#define NODE_VARS _nodeVars ? std::static_pointer_cast<NODE_VARS_TYPE>(_nodeVars) : std::shared_ptr< NODE_VARS_TYPE >()

#define NODE_DATA_TYPE _NODE_DATA_TYPE_1(NODE_TYPE, NODE_CLASS)
#define _NODE_DATA_TYPE_1(NODE_TYPE, NODE_CLASS) _NODE_DATA_TYPE_2(NODE_TYPE, NODE_CLASS)
#define _NODE_DATA_TYPE_2(NODE_TYPE, NODE_CLASS) NODE_TYPE##_##NODE_CLASS##_Data
#define NODE_DATA __nodeData ? std::static_pointer_cast<NODE_DATA_TYPE>(_nodeData) : std::shared_ptr< NODE_DATA_TYPE >()

#define PORT_PARAMS_SIZE(listID) (listID>=0 && listID<INPUT_PORT_NUM) ? _inputParams[listID].size() : 0
#define PORT_PARAMS_TYPE(listID) PORT_PARAMS_TYPE_##listID
#define PORT_PARAMS(listID, paramsID) (paramsID>=0 && paramsID<PORT_PARAMS_SIZE(listID) && _inputParams[listID].at(paramsID)) ? \
      std::static_pointer_cast< const PORT_PARAMS_TYPE(listID) >(_inputParams[listID].at(paramsID)) \
    : std::static_pointer_cast< const PORT_PARAMS_TYPE(listID) >()

#define PORT_DATA_SIZE(listID) (listID>=0 && listID<INPUT_PORT_NUM) ? _inputData[listID].size() : 0
#define PORT_DATA_TYPE(listID) PORT_DATA_TYPE_##listID
#define PORT_DATA(listID, dataID) (dataID>=0 && dataID<PORT_DATA_SIZE(listID) && _inputData[listID].at(dataID)) ? \
      std::static_pointer_cast< const PORT_DATA_TYPE(listID) >(_inputData[listID].at(dataID)) \
    : std::shared_ptr< const PORT_DATA_TYPE(listID) >()

//=================================================================================

#define ADD_PARAM(valueType, valueName, valueDefault) \
    private: valueType _init_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * params) \
    {if(!(xmlloader.getParamValue(#valueName,(NODE_PARAMS_TYPE*(params))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_PARAMS_TYPE*(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_init_##valueType##_##valueName##_Func();

#define ADD_ENUM_PARAM(valueType, valueName, valueDefault) \
    private: valueType _init_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * params) \
    {if(!(xmlloader.getEnumParamValue(#valueName,(NODE_PARAMS_TYPE*(params))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_PARAMS_TYPE*(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_init_##valueType##_##valueName##_Func();

#define ADD_UENUM_PARAM(valueType, valueName, valueDefault) \
    private: valueType _init_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * params) \
    {if(!(xmlloader.getUEnumParamValue(#valueName,(NODE_PARAMS_TYPE*(params))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_PARAMS_TYPE*(params))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_init_##valueType##_##valueName##_Func();

#define ADD_VAR(valueType, valueName, valueDefault) \
    private: valueType _init_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * vars) \
    {if(!(xmlloader.getParamValue(#valueName,(NODE_VARS_TYPE*(vars))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_VARS_TYPE*(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_init_##valueType##_##valueName##_Func();

#define ADD_ENUM_VAR(valueType, valueName, valueDefault) \
    private: valueType _init_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * vars) \
    {if(!(xmlloader.getEnumParamValue(#valueName,(NODE_VARS_TYPE*(vars))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_VARS_TYPE*(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_init_##valueType##_##valueName##_Func();

#define ADD_UENUM_VAR(valueType, valueName, valueDefault) \
    private: valueType _init_##valueType##_##valueName##_Func() \
    {_xmlloadfunclist.push_back([](XMLDomInterface & xmlloader, void * vars) \
    {if(!(xmlloader.getUEnumParamValue(#valueName,(NODE_VARS_TYPE*(vars))->valueName))) \
    {xmlloader.setParamDefault(#valueName,(NODE_VARS_TYPE*(vars))->valueName);}});return valueDefault;}; \
    public: valueType valueName=_init_##valueType##_##valueName##_Func();

#define ADD_INTERNAL_OBJECT_TRIGGER(triggerType, triggerName) \
    private: triggerType * _object_##triggerType##_##triggerName##_Func() \
    {triggerType * trigger=new triggerType; _objecttriggermap.insert(#triggerName, trigger); return trigger;}; \
    public: triggerType * triggerName=_object_##triggerType##_##triggerName##_Func();

#define ADD_INTERNAL_WIDGET_TRIGGER(triggerType, triggerName) \
    private: triggerType * _widget_##triggerType##_##triggerName##_Func() \
    {triggerType * trigger=new triggerType; _widgettriggermap.insert(#triggerName, trigger); _widgetmap.insert(#triggerName, trigger); return trigger;}; \
    public: triggerType * triggerName=_widget_##triggerType##_##triggerName##_Func();

#define ADD_INTERNAL_CONNECTION(triggerName,signalName,slotName, ...) \
    private: QString _connection_##triggerName##_##signalName##_##slotName##_Func() \
    {QString connection=QString("%1:%2").arg(SIGNAL(signalName(__VA_ARGS__))).arg(SLOT(slotName(__VA_ARGS__))); _connectionmap.insert(triggerName,connection); return connection;}; \
    private: QString _connection_##triggerName##_##signalName##_##slotName##_Qstr=_connection_##triggerName##_##signalName##_##slotName##_Func();

#define ADD_WIDGET(widgetType, widgetName) \
    private: widgetType * _widget_##widgetType##_##widgetName##_Func() \
    {widgetType * widget=new widgetType; _widgetmap.insert(#widgetName, widget); return widget;}; \
    public: widgetType * widgetName=_widget_##widgetType##_##widgetName##_Func();

#define ADD_LAYOUT(layoutType, layoutName) \
    private: layoutType * _layout_##layoutType##_##layoutName##_Func() \
    {layoutType * layout=new layoutType; _layoutmap.insert(#layoutName, layout); return layout;}; \
    public: layoutType * layoutName=_layout_##layoutType##_##layoutName##_Func()

//=================================================================================

#define NODE_FUNC_NAME(funcName) NODE_TYPE##_##NODE_CLASS##_##funcName
#define NODE_FUNC_DECL(returnType, funcName, ...) returnType NODE_FUNC_NAME(funcName)(ROBOTSDK_ARGS_DECL, __VA_ARGS__);
#define NODE_FUNC(returnType, funcName, ...) extern "C" Q_DECL_EXPORT returnType NODE_FUNC_NAME(funcName)(ROBOTSDK_ARGS_DECL, __VA_ARGS__)
#define NODE_FUNC_CALL(funcName, ...) NODE_FUNC_NAME(funcName)(ROBOTSDK_ARGS, __VA_ARGS__)

#define NODE_EXFUNC_NAME(funcName, exName) NODE_TYPE##_##NODE_CLASS##_##funcName##_##exName
#define NODE_EXFUNC_DECL(returnType, funcName, exName, ...) returnType NODE_EXFUNC_NAME(funcName, exName)(ROBOTSDK_ARGS_DECL, __VA_ARGS__);
#define NODE_EXFUNC(returnType, funcName, exName, ...) extern "C" Q_DECL_EXPORT returnType NODE_EXFUNC_NAME(funcName, exName)(ROBOTSDK_ARGS_DECL, __VA_ARGS__)
#define NODE_EXFUNC_CALL(funcName, exName, ...) NODE_EXFUNC_NAME(funcName, exName)(ROBOTSDK_ARGS, __VA_ARGS__)

//=================================================================================

#endif

}

#endif // DEFINES

