N,CameraSensor::camera,/home/alexanderhmw/SDK/RobotSDK_4.0/Modules/build-Sensor-Desktop_Qt_5_4_1_GCC_64bit-Release/libSensor.so,Config.xml,0,1
N,DPMDetector::detector, ,Config.xml,0,1
N,DPMModifier::modifier, ,Config.xml,2,1
N,DPMSampleSaver::saver, ,Config.xml,1,0
E,CameraSensor::camera,0,DPMModifier::modifier,0
E,DPMDetector::detector,0,DPMModifier::modifier,1
E,DPMModifier::modifier,0,DPMSampleSaver::saver,0
