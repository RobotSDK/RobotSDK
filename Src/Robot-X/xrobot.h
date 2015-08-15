#ifndef XROBOT_H
#define XROBOT_H

#include <QMainWindow>
#include <QGraphicsView>

#include"xgraph.h"
#include"xconfig.h"

namespace Ui {
class XRobot;
}

namespace RobotX
{

class GraphView : public QGraphicsView
{
    Q_OBJECT
public:
    GraphView(QWidget * parent=0);
signals:
    void signalHandleMenu();
protected:
    double ratio=1;
    bool ctrlflag=0;
protected:
    void mousePressEvent(QMouseEvent * event);
    void wheelEvent(QWheelEvent *event);
    void keyPressEvent(QKeyEvent *event);
    void keyReleaseEvent(QKeyEvent *event);
};

class XRobot : public QMainWindow
{
    Q_OBJECT
public:
    explicit XRobot(QWidget *parent = 0);
    ~XRobot();
private:
    Ui::XRobot *ui;
protected:
    GraphView * view;
    XGraph * graph;
protected:
    void closeEvent(QCloseEvent * event);
};

}

#endif // XROBOT_H
