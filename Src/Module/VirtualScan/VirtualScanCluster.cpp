#include"VirtualScanCluster.h"

//If you need use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VirtualScanGenerator)

//=================================================
//Original node functions

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
    auto params=NODE_PARAMS;
    auto outputdata=NODE_DATA;
    auto inputparams=PORT_PARAMS(0,0);
    auto data=PORT_DATA(0,0);

    double PI=3.141592654;
    double density=2*PI/inputparams->beamnum;

    outputdata->timestamp=data->timestamp;
    outputdata->virtualscan=data->virtualscan;
    outputdata->minheights=data->minheights;
    outputdata->maxheights=data->maxheights;
    outputdata->labels=data->labels;
    outputdata->labels.fill(0);
    outputdata->clusternum=0;
    outputdata->clusters.clear();

    uint i,n=outputdata->virtualscan.size();
    for(i=0;i<n;i++)
    {
        if(outputdata->virtualscan[i]>inputparams->minrange&&outputdata->labels[i]==0)
        {
            QQueue<uint> queue;
            queue.push_back(i);
            outputdata->clusternum++;
            outputdata->labels[i]=outputdata->clusternum;
            QVector<uint> records;
            while(!queue.isEmpty())
            {
                uint id=queue.front();
                queue.pop_front();
                records.push_back(id);
                double theta=asin(params->neighbordis/outputdata->virtualscan[id]);
                int neighbornum=int(theta/density+0.5);
                int j;
                for(j=-neighbornum;j<=neighbornum;j++)
                {
                    if(j==0)
                    {
                        continue;
                    }
                    int nid=int(id)+j;
                    nid>=0?nid:nid+n;
                    nid<n?nid:nid-n;
                    if(outputdata->virtualscan[nid]>0&&outputdata->labels[nid]==0)
                    {
                        double angle=fabs(j*density);
                        double distance=outputdata->virtualscan[nid]*outputdata->virtualscan[nid]
                                +outputdata->virtualscan[id]*outputdata->virtualscan[id]
                                -2*cos(angle)*outputdata->virtualscan[nid]*outputdata->virtualscan[id];
                        if(distance<=params->neighbordis)
                        {
                            outputdata->labels[nid]=outputdata->clusternum;
                            queue.push_back(nid);
                        }
                    }
                }
            }
            if(records.size()<params->minpointsnum)
            {
                int j,m=records.size();
                for(j=0;j<m;j++)
                {
                    outputdata->labels[records[j]]=0;
                }
                outputdata->clusternum--;
            }
        }
    }
    for(i=0;i<n;i++)
    {
        outputdata->clusters.insert(outputdata->labels[i],i);
    }
    return 1;
}
