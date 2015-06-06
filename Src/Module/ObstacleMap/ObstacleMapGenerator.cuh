#ifndef OBSTACLEMAPGENERATOR_CUH
#define OBSTACLEMAPGENERATOR_CUH

#include<opencv2/opencv.hpp>

extern "C" void cudaObstacleGenerator(int beamNum, const double * virtualScan, int mapSize, double gridSize, double obstacleFactor, u_char *map);

#endif // OBSTACLEMAPGENERATOR_CUH

