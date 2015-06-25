#include"ObstacleMapGenerator.cuh"

#include<cuda.h>
#include<cuda_runtime.h>

__global__ void kernelObstacleMapGenerator(int beamNum, double * px, double * py, int mapSize, double gridSize, double obstacleFactor, u_char * map, float * mapdata)
{
    int xid=blockIdx.x*blockDim.x+threadIdx.x;
    int yid=blockIdx.y*blockDim.y+threadIdx.y;
    int baseid=(mapSize*yid+xid)*3;
    int mapdataid=mapSize*yid+xid;
    if(xid<mapSize&&yid<mapSize)
    {
        int center=mapSize/2;
        if(xid!=center||yid!=center)
        {
            double x=(center-yid)*gridSize;
            double y=(center-xid)*gridSize;
            double PI=3.141592654;
            double density=2*PI/beamNum;
            double theta=atan2(y,x);
            if(theta<-PI)
            {
                theta=2*PI+theta;
            }
            double thetaid=(theta+PI)/density;
            int lid=(int(thetaid)+1)%beamNum;
            int rid=(int(thetaid))%beamNum;
            if((px[lid]==0&&py[lid]==0)||(px[rid]==0&&py[rid]==0))
            {
                map[baseid]=0;
                map[baseid+1]=255;
                map[baseid+2]=0;
                mapdata[mapdataid]=100;
            }
            else
            {
                double dx=px[lid]-px[rid];
                double dy=py[lid]-py[rid];
                double beta=(dx*py[lid]-px[lid]*dy)/(dx*y-x*dy);
                double dis=sqrt(x*x+y*y);
                double radius=obstacleFactor*gridSize;
                double delta=dis*(1-beta);
                if(delta<-radius) //free
                {
                    map[baseid]=0;
                    map[baseid+1]=255;
                    map[baseid+2]=0;
                    mapdata[mapdataid]=beta-1;
                }
                else if(delta>radius) //unkonw
                {
                    map[baseid]=0;
                    map[baseid+1]=0;
                    map[baseid+2]=255;
                    mapdata[mapdataid]=1-beta;
                }
                else //occ
                {
                    map[baseid]=255;
                    map[baseid+1]=0;
                    map[baseid+2]=0;
                    mapdata[mapdataid]=fabs(delta/radius);
                }
            }
        }
    }
}

void cudaObstacleMapGenerator(int beamNum, const double * virtualScan, int mapSize, double gridSize, double obstacleFactor, u_char * map, float *mapdata)
{
    size_t size=beamNum*sizeof(double);
    double * px=(double *)malloc(size);
    double * py=(double *)malloc(size);
    double PI=3.141592654;
    double density=2*PI/beamNum;
    int i;
    for(i=0;i<beamNum;i++)
    {
        double theta=i*density-PI;
        px[i]=virtualScan[i]*cos(theta);
        py[i]=virtualScan[i]*sin(theta);
    }
    double *dpx,*dpy;
    cudaMalloc((void**)(&dpx),size);
    cudaMalloc((void**)(&dpy),size);
    cudaMemcpy((void *)dpx,px,size,cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dpy,py,size,cudaMemcpyHostToDevice);

    u_char * dmap;
    float *dmapdata;
    size_t mapsize=mapSize*mapSize*3*sizeof(u_char);
    size_t mapdatasize=mapSize*mapSize*sizeof(float);
    cudaMalloc((void**)(&dmap),mapsize);
    cudaMemcpy((void *)dmap,map,mapsize,cudaMemcpyHostToDevice);
    cudaMalloc((void**)(&dmapdata),mapdatasize);
    cudaMemcpy((void *)dmapdata,mapdata,mapdatasize,cudaMemcpyHostToDevice);

    int dim=32;
    dim3 threadperblock(dim,dim);
    dim3 blockpergrid((mapSize+dim-1)/dim,(mapSize+dim-1)/dim);

    kernelObstacleMapGenerator<<<blockpergrid,threadperblock>>>(beamNum,dpx,dpy,mapSize,gridSize,obstacleFactor,dmap,dmapdata);

    cudaMemcpy(map,dmap,mapsize,cudaMemcpyDeviceToHost);
    cudaMemcpy(mapdata,dmapdata,mapdatasize,cudaMemcpyDeviceToHost);
    cudaFree(dpx);
    cudaFree(dpy);
    cudaFree(dmap);
    cudaFree(dmapdata);
}
