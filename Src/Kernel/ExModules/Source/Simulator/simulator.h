#ifndef SIMULATOR_H
#define SIMULATOR_H

/*! \defgroup Simulator_Library Simulator_Library
	\ingroup Source_Library
	\brief The Library of Simulator.
*/

/*! \addtogroup ExSource
	@{
*/

/*! \file simulator.h
	\brief Defines the class Simulator.
*/

#include<Modules/Source/source.h>
#include<qtimer.h>

/*! \class Simulator
	\brief Self-driven Simualtor derived from Source.
	\details
	- Simulator will be self-driven by timestamp in data file.
	- Overload Source::generateSourceDataSlot();
*/
class Simulator : public Source
{
	Q_OBJECT
public:
	/*! \fn Simulator(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QTime startTime, double simulateRate, QString qstrFuncEx=QString())
		\brief Constructor of SensorTimer.
		\param [in] qstrSharedLibrary The name of the shared library.
		\param [in] qstrNodeClass The class-name of the node.
		\param [in] qstrNodeName The node-name of the node.
		\param [in] qstrConfigName The name of the config file.
		\param [in] startTime The start time of the simulator.
		\param [in] simulateRate The simulate rate of the simulator.
		\param [in] qstrFuncEx The extension of Source::generateSourceData.
		\details
		- \a startTime: simulator will start from a data frame whose timestamp is just after \a startTime.
		- \a simulateRate: the interval between adjacent timestamps will be modified by simulateRate.
	*/
	Simulator(QString qstrSharedLibrary, QString qstrNodeClass, QString qstrNodeName, QString qstrConfigName, QTime startTime, double simulateRate, QString qstrFuncEx=QString());
protected:
	/*! \var starttime
		\brief The start time.
	*/
	QTime starttime;
	/*! \var startcurtime
		\brief The start current time for interval correctness.
	*/
	QTime startcurtime;
	/*! \var initialcurtime
		\brief The initial current time for interval correctness.
	*/
	QTime initialcurtime;
	/*! \var curtime
		\brief The current time.
	*/
	QTime curtime;
	/*! \var nexttime
		\brief The time next to current time.
	*/
	QTime nexttime;	
	/*! \var curoutputportindex
		\brief The current index of output port.
	*/
	QList<int> curoutputportindex;
	/*! \var nextoutputportindex
		\brief The index of output port next to current output port index.
	*/
	QList<int> nextoutputportindex;
	/*! \var curflag
		\brief The status of current generateSourceData.
	*/
	bool curflag;
	/*! \var nextflag
		\brief The status of next generateSourceData.
	*/
	bool nextflag;
	/*! \var curoutputdata
		\brief The current output data will be sent when the simulator is triggered 
	*/	
	boost::shared_ptr<void> curoutputdata;
	/*! \var nextoutputdata
		\brief Next output data will be stored to \ref curoutputdata after the \ref curoutputdata is sent and new data will be loaded. 
	*/
	boost::shared_ptr<void> nextoutputdata;
	/*! \var simulaterate
		\brief The simulate rate.
	*/
	double simulaterate;
	/*! \var simflag
		\brief The flag to indicate the simulator state.
	*/
	bool simflag;
public slots:
	/*! \fn void setStartTimeSlot(QTime startTime)
		\brief Set the start time.
		\param [in] startTime The start time.
	*/
	void setStartTimeSlot(QTime startTime);
	/*! \fn void setSimulateRateSlot(double simulateRate)
		\brief Set the simulator rate.
		\param [in] simulateRate The simulator rate.
	*/
	void setSimulateRateSlot(double simulateRate);
	/*! \fn void syncTimeTrackSlot()
		\brief Synchronize the timer track.
	*/
	void syncTimeTrackSlot();
	/*! \fn void resetTimeTrackSlot()
		\brief Reset the timer track.
		\details
		Reset
		- \ref curtime
		- \ref nexttime
		- \ref curoutputdata
		- \ref nextoutputdata
	*/
	void resetTimeTrackSlot();
	/*! \fn void startSimulatorSlot()
		\brief Start simulator.
	*/
	void startSimulatorSlot();
	/*! \fn void stopSimulatorSlot()
		\brief Stop simulator.
	*/
	void stopSimulatorSlot();
public slots:
	/*! \fn void generateSourceDataSlot()
		\brief Overload Source::generateSourceDataSlot() to realize self-driven.
		\details
		Each loop:
		- Triggered
		- Set timer to count (nexttime - curtime)
		- Send currentdata
		- Let currentdata = nextdata
		- Let curtime=nextime
		- Load new data to nextdata
		- Load new time to nexttime
	*/
	void generateSourceDataSlot();
};

/*! @}*/

#endif // SIMULATOR_H
