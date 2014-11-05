#ifndef USERINTERACT_H
#define USERINTERACT_H

/*! \defgroup UserInteract_Library UserInteract_Library
    \ingroup Source_Library
    \brief The Library of UserInteractMono.
*/

/*! \addtogroup ExSourceDrain
    @{
*/

/*! \file userinput.h
    \brief Defines the class UserInteractMono.
*/

#include<Modules/SourceDrain/sourcedrain.h>

/*! \class UserInteractMono
    \brief User Interact Mono Node derived from SourceDrainMono.
    \details
    - Provides 1 interface functions:
        - [private, optional] UserInteractMono::UIWidgets
    - Provides a UI interface
        - UserInteractMono::getUIWidgets()
*/
class UserInteractMono : public SourceDrainMono
{
    Q_OBJECT
public:
    /*! \fn UserInteractMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
        \brief The constructor of the class UserInteractMono. (For directly using)
        \param [in] qstrSharedLibrary The name of the shared library.
        \param [in] qstrNodeClass The class-name of the node.
        \param [in] qstrNodeName The node-name of the node.
        \param [in] qstrConfigName The name of the config file.
        \param [in] qstrFuncEx The extension of Source::generateSourceData.
        \details
        - Set the type-name as "VisualizationMono"
    */
    UserInteractMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
public:
    /*! \fn getUIWidgets()
        \brief Get the UI widgets.
        \return The UI widgets
    */
    QList<QWidget *> getUIWidgets();
protected:
    /*! \typedef void (*UIWidgetsFptr)(void * paramsPtr, void * varsPtr, QList<QWidget *> & widgets)
        \brief [optional] Function pointer type for interface function of getting node's UI widgets.
        \param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
        \param [in] varsPtr The node's variables(\ref Node::varsptr) that contains UI widget.
        \param [out] widgets The UI widgets.
    */
    typedef void (*UIWidgetsFptr)(void * paramsPtr, void * varsPtr, QList<QWidget *> & widgets);
    /*! \var UIWidgets
        \brief [private] Interface function of getting node's UI widget.
    */
    UIWidgetsFptr UIWidgets;
};

/*! \class UserInteractMulti
    \brief User Interact Multi Node derived from SourceDrainMulti.
    \details
    - Provides 1 interface functions:
        - [private, optional] UserInteractMulti::UIWidgets
    - Provides a UI interface
        - UserInteractMulti::getUIWidgets()
*/
class UserInteractMulti : public SourceDrainMulti
{
    Q_OBJECT
public:
    /*! \fn UserInteractMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
        \brief The constructor of the class UserInteractMulti. (For directly using)
        \param [in] qstrSharedLibrary The name of the shared library.
        \param [in] qstrNodeClass The class-name of the node.
        \param [in] qstrNodeName The node-name of the node.
        \param [in] qstrConfigName The name of the config file.
        \param [in] qstrFuncEx The extension of Source::generateSourceData.
        \details
        - Set the type-name as "VisualizationMulti"
    */
    UserInteractMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
public:
    /*! \fn getUIWidgets()
        \brief Get the UI widgets.
        \return The UI widgets
    */
    QList<QWidget *> getUIWidgets();
protected:
    /*! \typedef void (*UIWidgetsFptr)(void * paramsPtr, void * varsPtr, QList<QWidget *> & widgets)
        \brief [optional] Function pointer type for interface function of getting node's UI widgets.
        \param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
        \param [in] varsPtr The node's variables(\ref Node::varsptr) that contains UI widget.
        \param [out] widgets The UI widgets.
    */
    typedef void (*UIWidgetsFptr)(void * paramsPtr, void * varsPtr, QList<QWidget *> & widgets);
    /*! \var UIWidgets
        \brief [private] Interface function of getting node's UI widget.
    */
    UIWidgetsFptr UIWidgets;
};

/*! @}*/

#endif // USERINTERACT_H
