#include"VehicleTracker.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VehicleDetector)
PORT_DECL(1, ObstacleMapGlobalizer)


//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    STATE_TYPE(Vehicle) statemin={vars->state_x_min,vars->state_y_min,vars->state_theta_min,vars->state_v_min,vars->state_width_min,vars->state_length_min};
    STATE_TYPE(Vehicle) statemax={vars->state_x_max,vars->state_y_max,vars->state_theta_max,vars->state_v_max,vars->state_width_max,vars->state_length_max};
    STATE_TYPE(Vehicle) statemean={0,0,0,0,0,0};
    STATE_TYPE(Vehicle) statesigma={0,0,vars->state_theta_sigma,vars->state_v_sigma,0,0};
    PF_Vehicle_initialParticleFilter(vars->particlenum,statemin,statemax,statemean,statesigma);

    vars->mapsync.clear();

    vars->curtimestamp=QTime();
    vars->curtransform=cv::Mat::eye(4,4,CV_64F);
    vars->curtheta=0;
    vars->objectid.clear();
    vars->objectstate.clear();

    vars->measuredata.edgepointnum=vars->obmap_edgepointnum;
    vars->measuredata.margin=vars->obmap_margin;
    QStringList wtable=vars->obmap_wtable.split(",",QString::SkipEmptyParts);
    if(wtable.size!=3)
    {
        return 0;
    }
    vars->measuredata.wtable[0]=wtable[0].toFloat();
    vars->measuredata.wtable[1]=wtable[1].toFloat();
    vars->measuredata.wtable[2]=wtable[2].toFloat();

    vars->localheadvec=cv::Mat::zeros(4,1,CV_64F);
    vars->localheadvec.at<double>(0)=1;

	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;PORT_DECL(1, VehicleDetector)
    PF_Vehicle_clear();

    vars->mapsync.clear();

    vars->objectid.clear();
    vars->objectstate.clear();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    if(SYNC_START(vars->mapsync))
    {
        auto detection=SYNC_DATA(vars->mapsync,0);
        auto obstaclemap=SYNC_DATA(vars->mapsync,1);
        cv::Mat head=vars->curtransform*vars->localheadvec;
        if(vars->curtimestamp.isNull())
        {
            vars->curtimestamp=obstaclemap->timestamp;
            vars->curtransform=obstaclemap->transform;
            vars->curtheta=atan2(head.at<double>(1),head.at<double>(0));
            PF_Vehicle_addObjectStates(detection->objectids,detection->objectstates);
            return 0;
        }
        else
        {
            int deltamsec=vars->curtimestamp.msecsTo(obstaclemap->timestamp);
            vars->measuredata.gridsize=obstaclemap->gridsize;
            vars->measuredata.mapsize=obstaclemap->map.rows;
            vars->measuredata.map=obstaclemap->map.data;
            float dx=obstaclemap->transform.at<double>(0,3)-vars->curtransform.at<double>(0,3);
            float dy=obstaclemap->transform.at<double>(1,3)-vars->curtransform.at<double>(1,3);
            float theta1=vars->curtheta;
            float theta2=atan2(head.at<double>(1),head.at<double>(0));
            PF_Vehicle_advanceParticleFilter2D(deltamsec,vars->measuredata,dx,dy,theta1,theta2);
            PF_Vehicle_removeParticles(vars->threshold);
            PF_Vehicle_estimateObjects(vars->objectid,vars->objectstate);

            vars->curtimestamp=obstaclemap->timestamp;
            vars->curtransform=obstaclemap->transform;
            vars->curtheta=theta2;
            PF_Vehicle_addObjectStates(detection->objectids,detection->objectstates);

            auto data=NODE_DATA;
            data->transform=vars->curtransform.clone();
            data->objectid=vars->objectid;
            data->objectstate=vars->objectstate;
            return 1;
        }
    }
    return 0;
}
