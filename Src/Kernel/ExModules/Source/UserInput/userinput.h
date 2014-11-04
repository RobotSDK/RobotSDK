#ifndef USERINPUT_H
#define USERINPUT_H

/*! \defgroup UserInput_Library UserInput_Library
	\ingroup Source_Library
	\brief The Library of UserInput.
*/

/*! \addtogroup ExSource
	@{
*/

/*! \file userinput.h
	\brief Defines the class UserInput.
*/

#include<Modules/Source/source.h>

/*! \class UserInput
	\brief User Input Node derived from Source.
	\details
	- Provides 1 interface functions:
		- [private, optional] UserInput::UIWidgets
	- Provides a UI interface
		- UserInput::getUIWidgets()
*/
class UserInput : public Source
{
	Q_OBJECT
public:
	/*! \fn UserInput(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class UserInput. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of Source::generateSourceData.
		\details
		- Set the type-name as "VisualizationMono"
	*/
	UserInput(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
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

#endif // USERINPUT_H
