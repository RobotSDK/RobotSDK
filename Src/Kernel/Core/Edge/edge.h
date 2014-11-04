#ifndef EDGE_H
#define EDGE_H

/*! \class QWidget
    \brief QWidget in Qt.
*/

/*! \defgroup Edge Edge
    \ingroup Core
    \brief The edge of the graph model.
*/

/*! \addtogroup Edge
    @{
*/

/*! \file edge.h
    \brief Defines the class Edge.
    \details Edge in graph model.
*/


#include<qwidget.h>
#include<qlayout.h>
#include<qtablewidget.h>
#include<qlineedit.h>
#include<qheaderview.h>
#include<qset.h>
#include<qlist.h>
#include<qvector.h>
//#include<qstring.h>
#include<qthread.h>
#include<qtimer.h>
#include<qscrollarea.h>
#include<qsplitter.h>

#include<Core/Node/node.h>

#include"triggerlog.h"
#include"triggerview.h"

/*! \class Edge
    \brief Class Edge automatically connects its all nodes and throw them into seperated threads.
    \details
    - Automatically manages all nodes. (Including deleteLater)
    - Automatically manages all threads for nodes. (Including start(), exit() and wait())
    - Usage:
        - Edge::addNode(Node * node);
            - Add node.
            - Throw node to its thread.
            - Connect to Node::openNodeSlot() and Node::closeNodeSlot() in blockedqueue mode.
        - Edge::connectAll();
            - Automatically connect nodes using their Node::inputnodesname and Node::outputnodesname.
        - Edge::openAllNodesSlot();
            - Open all nodes.
        - Edge::closeAllNodesSlot();
            - Close all nodes.
        - Edge::clear();
            - Close all nodes.
            - Exit all threads.
            - Disconnect all connections.
            - Delete nodes.
            - Delete threads.
    - Remarks
        - An application with Edge needs not to manage nodes and threads.
        - Only needs to create and config nodes.
        - Edge::clear() is not required to explicitly call in the destructor.
*/
class Edge : public QWidget
{
    Q_OBJECT
public:
    /*! \fn Edge()
        \brief Constructor of Edge.
    */
    Edge();
    /*! \fn ~Edge()
        \brief Destructor of Edge.
    */
    ~Edge();
protected:
    /*! \var nodepool
        \brief The pool of all nodes.
    */
    QSet<Node *> nodepool;
    /*! \var threads
        \brief Threads for nodes.
    */
    QVector<QThread *> threads;
    /*! \var timerspeed
        \brief Input the timer speed (ms).
    */
    QLineEdit timerspeed;
    /*! \var timer
        \brief Timer to drive TriggerLog and TriggerView.
    */
    QTimer timer;
    /*! playpause
        \brief The button to play or pause the monitor.
    */
    QPushButton playpause;
    /*! \var timerangeinput
        \brief Input the time range for TriggerView.
    */
    QLineEdit timerangeinput;
	/*! \var zoomratioinput
        \brief Input the zoom ratio for TriggerView.
    */
	QLineEdit zoomratioinput;
    /*! \var timerange
        \brief The time range to show.
    */
    int timerange;
    /*! \var timeintervalinput
        \brief Input the time interval for TriggerView.
    */
    QLineEdit timeintervalinput;
    /*! \var timeinterval
        \brief The time interval to show.
    */
    int timeinterval;
	/*! \var zoomratio
        \brief The zoom ratio.
    */
    double zoomratio;
	/*! \var panel
        \brief panel to contain \ref TriggerLog and \ref TriggerView.
    */
	QGridLayout panel;
protected:
    /*! \fn bool connectNodes(Node * inputNode, Node * outputNode)
        \brief Connect two nodes.
        \param [in] inputNode input node.
        \param [in] outputNode output node.
        \return 1 for success and 0 for failure.
    */
    bool connectNodes(Node * inputNode, Node * outputNode);
    /*! \fn bool disconnectNodes(Node * inputNode, Node * outputNode)
        \brief Disconnect two nodes.
        \param [in] inputNode input node.
        \param [in] outputNode output node.
        \return 1 for success and 0 for failure.
    */
    bool disconnectNodes(Node * inputNode, Node * outputNode);
public:
    /*! \fn void addNode(Node * node, bool gotoThread=1, bool needMonitor=1)
        \brief Add node into \ref nodepool.
        \param [in] node The added node.
        \param [in] gotoThread The flag to determine whether the node goes to new thead.
        \param [in] needMonitor The flag to determine whether the node needs monitor.
    */
    void addNode(Node * node, bool gotoThread=1, bool needMonitor=1);
    /*! \fn void clear()
        \brief Clear all nodes and their threads.
    */
    void clear();
    /*! \fn bool connectAll()
        \brief Connect all nodes.
    */
    bool connectAll();
protected:
    /*! \fn bool disconnectAll()
        \brief Disconnect all nodes.
    */
    bool disconnectAll();
public slots:
    /*! \fn void openAllNodesSlot()
        \brief Slot function for opening all nodes.
    */
    void openAllNodesSlot();
    /*! \fn void closeAllNodesSlot()
        \brief Slot function for closing all nodes.
    */
    void closeAllNodesSlot();
    /*! \fn void playPauseTimerSlot()
        \brief Slot function to play or pause monitor,
    */
    void playPauseTimerSlot();
    /*! \fn void setTimeLineSlot();
        \brief Slot function to set time line appearance.
    */
    void setTimeLineSlot();
    /*! \fn void drawSlot()
        \brief Slot function triggered by \ref timer to draw monitor. (Still has bug)
    */
    void drawSlot();
signals:
    /*! \fn void openAllNodesSignal()
        \brief Signal for opening all nodes.
    */
    void openAllNodesSignal();
    /*! \fn void closeAllNodesSignal()
        \brief Signal for closing all nodes.
    */
    void closeAllNodesSignal();
    /*! \fn void drawSignal(QDateTime curDateTime)
        \brief Signal for drawing monitor.
        \param [in] curDateTime Current date and time.
    */
    void drawSignal(QDateTime curDateTime);
};

/*! @}*/

#endif // EDGE_H
