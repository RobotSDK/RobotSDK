#ifndef TRIGGERLOG_H
#define TRIGGERLOG_H

/*! \addtogroup Edge
    @{
*/

/*! \file triggerlog.h
    \brief Defines the TriggerLog.
    \details Log to show openning, processing and closing.
*/

#include<qwidget.h>
#include<qpushbutton.h>
#include<qplaintextedit.h>
#include<qlayout.h>
#include<qfiledialog.h>
#include<qstring.h>
#include<qfile.h>

#include<Core/Node/node.h>

/*! \class TriggerLog
    \brief Class TriggerLog shows the log of the Node's status and can save the log to file.
*/
class TriggerLog : public QWidget
{
    Q_OBJECT
public:
    /*! \fn TriggerLog(QWidget *parent, Node * node, bool gotoThread)
        \brief The constructor of the TriggerLog.
        \param [in] parent The parent widget, it is Edge.
        \param [in] node The node to be monitored.
        \param [in] gotoThread The flag to show whether the node will be moved to sub-thread.
    */
    TriggerLog(QWidget *parent, Node * node, bool gotoThread);
    /*! \fn ~TriggerLog()
        \brief The destructor of the TriggerLog.
    */
    ~TriggerLog();
protected:
    /*! \var backupnode
        \brief To store the node for disconnection.
    */
    Node * backupnode;
    /*! \var nodetag
        \brief The tag of the node (Type_Class_Name)
    */
    QString nodetag;
    /*! \var log
        \brief To show the info.
    */
    QPlainTextEdit log;
public slots:
    /*! \fn void saveSlot()
        \brief Slot function to save \ref log.
    */
    void saveSlot();
    /*! \fn void clearSlot()
        \brief Slot function to clear \ref log.
    */
    void clearSlot();
public slots:
    /*! \fn void openNodeSlot()
        \brief Slot function to receive openNodeSignal() from Node.
    */
    void openNodeSlot();
    /*! \fn void openNodeErrorSlot()
        \brief Slot function to receive openNodeErrorSignal() from Node.
    */
    void openNodeErrorSlot();
    /*! \fn void closeNodeSlot()
        \brief Slot function to receive closeNodeSignal() from Node.
    */
    void closeNodeSlot();
    /*! \fn void closeNodeErrorSlot()
        \brief Slot function to receive closeNodeErrorSignal() from Node.
    */
    void closeNodeErrorSlot();
    /*! \fn void nodeTriggerTimeSlot(QDateTime curDateTime, Node::NodeTriggerState nodeTriggerState)
        \brief Slot function to receive trigger signal from Node.
        \param [in] curDateTime Current Date and Time.
        \param [in] nodeTriggerState The state of the node's trigger.
    */
    void nodeTriggerTimeSlot(QDateTime curDateTime, Node::NodeTriggerState nodeTriggerState);
};

/*! @}*/

#endif // TRIGGERLOG_H
