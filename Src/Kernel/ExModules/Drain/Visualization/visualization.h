#ifndef VISUALIZATION_H
#define VISUALIZATION_H

/*! \defgroup VisualizationMono_Library VisualizationMono_Library
	\ingroup DrainMono_Library
	\brief The Library of VisualizationMono.
*/

/*! \defgroup VisualizationMulti_Library VisualizationMulti_Library
	\ingroup DrainMulti_Library
	\brief The Library of VisualizationMulti.
*/

/*! \addtogroup ExDrain
	@{
*/

/*! \file visualization.h
	\brief Defines the class VisualizationMono and VisualizationMulti.
*/

#include<Modules/Drain/drain.h>

/*! \class VisualizationMono
	\brief VisualizationMono visualize mono drain data.
	\details
	- Provides 1 interface function:
		- [private, optional] VisualizationMono::visualizationWidget
	- Provides a visualization interface
		- VisualizationMono::getVisualizationWidget()
	- Provides a slot function:
		- resetVisualizationSlot()
*/
class VisualizationMono : public DrainMono
{
	Q_OBJECT
public:
	/*! \fn VisualizationMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class VisualizationMono. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMono::processMonoDrainData.
		\details
		- Set the type-name as "VisualizationMono"
	*/
	VisualizationMono(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
public:
	/*! \fn QList<QWidget *> getVisualizationWidgets()
		\brief Get the visualization widgets.
		\return The visualization widgets
	*/
	QList<QWidget *> getVisualizationWidgets();
public slots:
	/*! \fn void resetVisualizationSlot()
		\brief Reset the visualization widgets.
	*/
	void resetVisualizationSlot();
protected:
	/*! \typedef void (*visualizationWidgetsFptr)(void * paramsPtr, void * varsPtr, QList<QWidget *> & widgets)
		\brief [optional] Function pointer type for interface function of getting node's visualization widgets.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr) that contains visualization widget.
		\param [out] widgets The visualization widgets.
	*/
	typedef void (*visualizationWidgetsFptr)(void * paramsPtr, void * varsPtr, QList<QWidget *> & widgets);
	/*! \var visualizationWidgets
		\brief [private] Interface function of getting node's visualization widget.
	*/
	visualizationWidgetsFptr visualizationWidgets;
};

/*! \class VisualizationMulti
	\brief VisualizationMulti visualize multi drain data.
	\details
	- Provides 1 interface functions:
		- [private, optional] VisualizationMulti::getVisualizationWidget
	- Provides a visualization interface
		- VisualizationMulti::visualizationWidget()
	- Provides a slot function:
		- resetVisualizationSlot()
*/
class VisualizationMulti : public DrainMulti
{
	Q_OBJECT
public:
	/*! \fn VisualizationMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief The constructor of the class VisualizationMulti. (For directly using)
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of DrainMulti::processMultiDrainData.
		\details
		- Set the type-name as "VisualizationMulti"
	*/
	VisualizationMulti(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
public:
	/*! \fn getVisualizationWidgets()
		\brief Get the visualization widget.
		\return The visualization widget
	*/
	QList<QWidget *> getVisualizationWidgets();
public slots:
	/*! \fn void resetVisualizationSlot()
		\brief Reset the visualization widgets.
	*/
	void resetVisualizationSlot();
protected:
	/*! \typedef void (*visualizationWidgetsFptr)(void * paramsPtr, void * varsPtr, QList<QWidget *> & widgets)
		\brief [optional] Function pointer type for interface function of getting node's visualization widgets.
		\param [in] paramsPtr The node's parameters(\ref Node::paramsptr).
		\param [in] varsPtr The node's variables(\ref Node::varsptr) that contains visualization widget.
		\param [out] widgets The visualization widgets.
	*/
	typedef void (*visualizationWidgetsFptr)(void * paramsPtr, void * varsPtr, QList<QWidget *> & widgets);
	/*! \var visualizationWidgets
		\brief [private] Interface function of getting node's visualization widget.
	*/
	visualizationWidgetsFptr visualizationWidgets;
};

/*! @}*/

#endif // VISUALIZATION_H
