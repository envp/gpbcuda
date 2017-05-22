#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

//#include "src/kernels/filters.cu"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    if( argc != 3 )
    {
        cout << "Usage: ./gpbcuda <InputImagePath> <OutputImagePath>" << endl;
        return 1;
    }

    Mat srcImg = imread( argv[1], 1 );
    cout << "GPU Count:" << getGPUCount() << endl;
    imwrite( argv[2], srcImg);
}
