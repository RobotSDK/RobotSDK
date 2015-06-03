#include"ObstacleMapGenerator.cuh"

#include<cuda.h>
#include<cuda_runtime.h>

__global__ void kernelObstacleMapGenerator(int beamNum, double * px, double * py, int mapSize, double gridSize, u_char * map)
{
    int xid=blockIdx.x*blockDim.x+threadIdx.x;
    int yid=blockIdx.y*blockDim.y+threadIdx.y;
    int baseid=(mapSize*yid+xid)*3;
    if(xid<mapSize&&yid<mapSize)
    {
        int center=mapSize/2+1;
        if(xid!=center||yid!=center)
        {
            double x=(xid-center)*gridSize;
            double y=(center-yid)*gridSize;
            double theta=atan2(y,x);
            double PI=3.141592654;
            double density=2*PI/beamNum;
            double thetaid=(theta+PI/2)/density;
            int lid=(int(thetaid)+1)%beamNum;
            int rid=(int(thetaid))%beamNum;
            if((px[lid]==0&&py[lid]==0)||(px[rid]==0&&py[rid]==0))
            {
                map[baseid]=0;
                map[baseid+1]=255;
                map[baseid+2]=0;
            }
            else
            {
                double dx=px[lid]-px[rid];
                double dy=py[lid]-py[rid];
                double beta=(dx*py[lid]-px[lid]*dy)/(dx*y-x*dy);
                double dis=sqrt(x*x+y*y);
                double radius=1.414213562*gridSize;
                double delta=dis*(1-beta);
                if(delta<-radius)
                {
                    map[baseid]=0;
                    map[baseid+1]=255;
                    map[baseid+2]=0;
                }
                else if(delta>radius)
                {
                    map[baseid]=255;
                    map[baseid+1]=0;
                    map[baseid+2]=0;
                }
                else
                {
                    map[baseid]=0;
                    map[baseid+1]=0;
                    map[baseid+2]=255;
                }
            }
        }
    }
}

void cudaObstacleGenerator(int beamNum, const double * virtualScan, int mapSize, double gridSize, u_char * map)
{
    size_t size=beamNum*sizeof(double);
    double * px=(double *)malloc(size);
    double * py=(double *)malloc(size);
    double PI=3.141592654;
    double density=2*PI/beamNum;
    int i;
    for(i=0;i<beamNum;i++)
    {
        double theta=i*density+PI/2;
        px[i]=virtualScan[i]*cos(theta);
        py[i]=virtualScan[i]*sin(theta);
    }
    double *dpx,*dpy;
    cudaMalloc((void**)(&dpx),size);
    cudaMalloc((void**)(&dpy),size);
    cudaMemcpy((void *)dpx,px,size,cudaMemcpyHostToDevice);
    cudaMemcpy((void *)dpy,py,size,cudaMemcpyHostToDevice);

    u_char * dmap;
    size_t mapsize=mapSize*mapSize*3*sizeof(u_char);
    cudaMalloc((void**)(&dmap),mapsize);
    cudaMemcpy((void *)dmap,map,mapsize,cudaMemcpyHostToDevice);

    int dim=32;
    dim3 threadperblock(dim,dim);
    dim3 blockpergrid((mapSize+dim-1)/dim,(mapSize+dim-1)/dim);

    kernelObstacleMapGenerator<<<blockpergrid,threadperblock>>>(beamNum,dpx,dpy,mapSize,gridSize,dmap);

    cudaMemcpy(map,dmap,mapsize,cudaMemcpyDeviceToHost);
    cudaFree(dpx);
    cudaFree(dpy);
    cudaFree(dmap);
}
