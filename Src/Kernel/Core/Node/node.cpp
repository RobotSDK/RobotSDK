#include"node.h"

using namespace RobotSDK;

Node::Node(QString nodeClass, QString nodeName)
    : QObject(NULL)
{
    _nodeclass=nodeClass;
    _nodename=nodeName;
}

void Node::slotInitialNode()
{

}
