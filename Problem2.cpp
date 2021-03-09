#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
using namespace cv;
int main( int argc, char** argv )
{
	// Declare the variables we are going to use
	Mat src, src_gray, dst;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	const char* window_name = "Laplace Demo";	
	const char* imageName = argc >=2 ? argv[1] : "lena.jpg";
    //src = imread( samples::findFile( imageName ), IMREAD_COLOR ); // Load an image
    // Check if image is loaded fine
    //if(src.empty()){
    //    printf(" Error opening image\n");
    //    printf(" Program Arguments: [image_name -- default lena.jpg] \n");
    //    return -1;
    //}
	N = 51
	float uinit[N][N] = {0};
	float utinit[N][N] = {0};

	uinit[N/2][N/2] = 10;

	float Ucv = uinit
	float Utcv = utinit

	
	// Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
	//GaussianBlur( src, src, Size(3, 3), 0, 0, BORDER_DEFAULT );
	//cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to grayscale
	Mat src(N,N, CV_32F,Scalar::all(0))
	Mat abs_dst;
	Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
	return 0;
}
