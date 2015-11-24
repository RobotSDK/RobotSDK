#include"LineExtractor.h"
using namespace RobotSDK_Module;

//If you need to use extended node, please uncomment below and comment the using of default node
//USE_EXTENDED_NODE(ExtendedNodeClass[,...])
USE_DEFAULT_NODE

//=================================================
//Uncomment below PORT_DECL and set input node class name
PORT_DECL(0, VirtualScanGenerator)
PORT_DECL(1, VirtualScanROI_DPM)

//=================================================
//Original node functions

//If you don't need to manually open node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, openNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->sync.clear();
	return 1;
}

//If you don't need to manually close node, you can delete this code segment
NODE_FUNC_DEF_EXPORT(bool, closeNode)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    vars->sync.clear();
	return 1;
}

//This is original main function, you must keep it
NODE_FUNC_DEF_EXPORT(bool, main)
{
	NOUNUSEDWARNING;
    auto vars=NODE_VARS;
    auto params=NODE_PARAMS;
    if(SYNC_START(vars->sync))
    {
        auto vscan=SYNC_DATA(vars->sync,0);
        auto roi=SYNC_DATA(vars->sync,1);
        auto data=NODE_DATA;
        double PI=3.14159265359;
        int beamnum=vscan->virtualscan.size();
        double density=2*PI/beamnum;
        if(roi->roi.size()==0)
        {
            return 0;
            data->timestamp=vscan->timestamp;
            ExtractedLines extractedline;
            mrpt::math::CVectorDouble x;
            mrpt::math::CVectorDouble y;
            extractedline.id=-1;
            extractedline.starttheta=0;
            extractedline.endtheta=2*PI;
            for(int i=0;i<beamnum;i++)
            {
                if(vscan->virtualscan[i]<=0)
                {
                    continue;
                }
                double theta=i*density-PI;
                QPointF point(vscan->virtualscan[i]*cos(theta),vscan->virtualscan[i]*sin(theta));
                x.push_back(point.x());
                y.push_back(point.y());
                extractedline.points.push_back(point);
            }
            std::vector<std::pair<size_t,mrpt::math::TLine2D > > detectedLines;
            mrpt::math::ransac_detect_2D_lines(x,y,detectedLines,params->dist_threshold,params->min_inliers);
            for(int i=0;i<detectedLines.size();i++)
            {
                mrpt::math::TLine2D tline2d=detectedLines[i].second;
                if(tline2d.coefs[1]!=0)
                {
                    double px,py;
                    px=-100;py=-(tline2d.coefs[0]*px+tline2d.coefs[2])/tline2d.coefs[1];
                    QPointF p1(px,py);
                    px=100;py=-(tline2d.coefs[0]*px+tline2d.coefs[2])/tline2d.coefs[1];
                    QPointF p2(px,py);
                    extractedline.lines.push_back(QLineF(p1,p2));
                }
                else
                {
                    double px,py;
                    py=-100;px=-(tline2d.coefs[0]*py+tline2d.coefs[2])/tline2d.coefs[1];
                    QPointF p1(px,py);
                    py=100;px=-(tline2d.coefs[0]*py+tline2d.coefs[2])/tline2d.coefs[1];
                    QPointF p2(px,py);
                    extractedline.lines.push_back(QLineF(p1,p2));
                }
            }
            data->lines.push_back(extractedline);
            return 1;
        }
        else
        {
            data->timestamp=vscan->timestamp;
            for(int k=0;k<roi->roi.size();k++)
            {
                ExtractedLines extractedline;
                mrpt::math::CVectorDouble x;
                mrpt::math::CVectorDouble y;
                extractedline.id=k;
                extractedline.starttheta=roi->roi[k].first*density-PI;
                extractedline.endtheta=roi->roi[k].second*density-PI;
                for(int i=roi->roi[k].first;i<=roi->roi[k].second;i++)
                {
                    int beamid=i%beamnum;
                    if(vscan->virtualscan[beamid]<=0)
                    {
                        continue;
                    }
                    double theta=beamid*density-PI;
                    QPointF point(vscan->virtualscan[beamid]*cos(theta),vscan->virtualscan[beamid]*sin(theta));
                    x.push_back(point.x());
                    y.push_back(point.y());
                    extractedline.points.push_back(point);
                }
                std::vector<std::pair<size_t,mrpt::math::TLine2D > > detectedLines;
                mrpt::math::ransac_detect_2D_lines(x,y,detectedLines,params->dist_threshold,params->min_inliers);
                for(int i=0;i<detectedLines.size();i++)
                {
                    mrpt::math::TLine2D tline2d=detectedLines[i].second;
                    if(tline2d.coefs[1]!=0)
                    {
                        double px,py;
                        px=-100;py=-(tline2d.coefs[0]*px+tline2d.coefs[2])/tline2d.coefs[1];
                        QPointF p1(px,py);
                        px=100;py=-(tline2d.coefs[0]*px+tline2d.coefs[2])/tline2d.coefs[1];
                        QPointF p2(px,py);
                        extractedline.lines.push_back(QLineF(p1,p2));
                    }
                    else
                    {
                        double px,py;
                        py=-100;px=-(tline2d.coefs[0]*py+tline2d.coefs[2])/tline2d.coefs[1];
                        QPointF p1(px,py);
                        py=100;px=-(tline2d.coefs[0]*py+tline2d.coefs[2])/tline2d.coefs[1];
                        QPointF p2(px,py);
                        extractedline.lines.push_back(QLineF(p1,p2));
                    }
                }
                data->lines.push_back(extractedline);
            }
            return 1;
        }
    }
    return 0;
}
