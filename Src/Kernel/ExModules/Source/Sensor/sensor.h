#ifndef SENSOR_H
#define SENSOR_H

/*! \defgroup SensorTimer_Library SensorTimer_Library
	\ingroup Source_Library
	\brief The Library of SensorTimer.
*/

/*! \defgroup SensorExternalEvent_Library SensorExternalEvent_Library
	\ingroup Source_Library
	\brief The Library of SensorExternalEvent.
*/

/*! \defgroup SensorInternalEvent_Library SensorInternalEvent_Library
	\ingroup Source_Library
	\brief The Library of SensorInternalEvent.
*/

/*! \addtogroup ExSource
	@{
*/

/*! \file sensor.h
	\brief Defines the class SensorTimer, SensorExternalEvent and SensorInternalEvent.
*/

#include<Modules/Source/source.h>
#include<qtimer.h>

/*! \class SensorTimer
	\brief Timer driven sensor.
	\details
	- Driven by QTimer.
	- Provides 3 slot functions to control timer:
		- SensorTimer::setTimerSlot(int msec)
		- SensorTimer::startTimerSlot()
		- SensorTimer::stopTimerSlot()
	- Overload Node::closeNodeSlot():
		- SensorTimer::closeNodeSlot();
*/
class SensorTimer : public Source
{
	Q_OBJECT
public:
	/*! \fn SensorTimer(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, int msec, QString qstrFuncEx=QString())
		\brief Constructor of SensorTimer.
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] msec The millisecond of the interval.
		\param [in] qstrFuncEx The extension of Source::generateSourceData.
	*/
	SensorTimer(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, int msec, QString qstrFuncEx=QString());
	/*! \fn ~SensorTimer()
		\brief Destructor of SensorTimer. Stop \ref timerthread.
	*/
	~SensorTimer();
protected:
	/*! \var timer
		\brief The timer to drive sensor.
	*/
	QTimer timer;
	/*! \var timerthread
		\brief The thread for timer.
	*/
	QThread timerthread;
public slots:
	/*! \fn void setTimerSlot(int msec)
		\brief Set timer's interval to \a msec.
		\param [in] msec Millisecond of the interval.
	*/
	void setTimerSlot(int msec);
	/*! \fn void startTimerSlot()
		\brief Start timer.
	*/
	void startTimerSlot();
	/*! \fn void stopTimerSlot()
		\brief Stop timer.
	*/
	void stopTimerSlot();
	/*! \fn void closeNodeSlot()
		\brief The slot function for closing node.
	*/
	void closeNodeSlot();
signals:
	/*! \fn void startTimerSignal()
		\brief Signal to start \ref timer.
	*/
	void startTimerSignal();
	/*! \fn void startTimerSignal(int msec)
		\brief Signal to set interval and restart \ref timer.
		\param [in] msec Millisecond of the interval.
	*/
	void startTimerSignal(int msec);
	/*! \fn void stopTimerSignal()
		\brief Signal to stop \ref timer.
	*/
	void stopTimerSignal();	
};

/*! \class SensorExternalEvent
	\brief External event driven sensor.
	\details
	- Driven by external event, such as push button signal.
*/
class SensorExternalEvent : public Source
{
	Q_OBJECT
public:
	/*! \fn SensorExternalEvent(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QObject * externalTrigger, const char * externalTriggerSignal, QString qstrFuncEx=QString())
		\brief Constructor of SensorExternalEvent.
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] externalTrigger The external trigger.
		\param [in] externalTriggerSignal The signal of external trigger.
		\param [in] qstrFuncEx The extension of Source::generateSourceData.
	*/
	SensorExternalEvent(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QObject * externalTrigger, const char * externalTriggerSignal, QString qstrFuncEx=QString());
};

/*! \class SensorInternalEvent
	\brief Internal event driven sensor.
	\details
	- Driven by internal event stored in \ref Node::varsptr.
*/
class SensorInternalEvent : public Source
{
	Q_OBJECT
public:
	/*! \fn SensorInternalEvent(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString())
		\brief Constructor of SensorInternalEvent.
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] qstrFuncEx The extension of Source::generateSourceData.
		\details
		It will automatically connect internal trigger. \sa Node::getInternalTrigger.
	*/
	SensorInternalEvent(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QString qstrFuncEx=QString());
};

/*! @}*/

#endif // SENSOR_H
