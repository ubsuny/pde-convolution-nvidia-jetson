#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <string>
#include <cstdlib>
#include <iostream>
#include <cstdio>
#include <ctime>

int main(int argc, char** argv){

	//create matrices
	int N = 51;
	
	cv::Mat U = cv::Mat::zeros(N, N, CV_32F);
	cv::Mat Ut = cv::Mat::zeros(N, N, CV_32F);
	cv::Mat lU = cv::Mat::zeros(N, N, CV_32F);

	
	U.at<float>(N/2,N/2) = 10;	
	std::cout << U << std::endl;	

	//double lap_vals[3][3] = {{0, 1, 0},{1, -4, 1},{0, 1, 0}};
	cv::Mat LaPlace = cv::Mat::zeros(3, 3, CV_32F);
	LaPlace.at<float>(0,1) = 1;
	LaPlace.at<float>(1,0) = 1;
	LaPlace.at<float>(1,1) = -4;
	LaPlace.at<float>(1,2) = 1;
	LaPlace.at<float>(2,1) = 1;

	std::cout << LaPlace <<std::endl;



	std::clock_t start;
	double duration;
	start = std::clock();


	/*Algorithm*/
	cv::Mat U_out;
	cv::Ptr<cv::cuda::Filter> filter2D;
	filter2D = cv::cuda::createLinearFilter(U.type(), -1, LaPlace);
	for (int k = 0; k < 50; ++k){
		//lU = cv::cuda::Convolution::convolve(U, LaPlace, U_out, bool ccorr = false, Stream& stream = Stream::Null());
		filter2D->apply(U, Ut);
		U = U + Ut;
		Ut = Ut + (1/4)*lU;
	
	} 

	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout<<"printf: "<< duration <<'\n';
    
	
	//load the image
    /*
    image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(image, image_gray, CV_BGR2GRAY);
    std::cout<<"Original image size: "<<image.size()<<std::endl;

    // generate gabor kernels
    double sigma = 0.5*lambda;
    cv::Mat sin_gabor = cv::getGaborKernel(cv::Size(), sigma, theta*M_PI/180.0, lambda, 1.0, M_PI/2.0, CV_32F);
    cv::Mat cos_gabor = cv::getGaborKernel(cv::Size(), sigma, theta*M_PI/180.0, lambda, 1.0, 0.0, CV_32F);
    std::cout<<"Kernel size: "<<sin_gabor.size()<<std::endl;

    // get filter responses (CPU)
    image_gray.convertTo(image_f, CV_32F, 1.0/256.0);
    cv::filter2D(image_f,sin_response, -1, sin_gabor, cv::Point(-1,-1));
    cv::filter2D(image_f,cos_response, -1, cos_gabor, cv::Point(-1,-1));
    cv::multiply(sin_response, sin_response, sin_response);
    cv::multiply(cos_response, cos_response, cos_response);
    
    // get filter responses (GPU)
    cv::gpu::GpuMat image_d, image_f_d, sin_gabor_d, cos_gabor_d, padded_image_d;
    cv::gpu::GpuMat sin_response_d, cos_response_d;

    image_d.upload(image_gray);
    image_d.convertTo(image_f_d, CV_32F, 1.0/256.0);
    sin_gabor_d.upload(sin_gabor);
    cos_gabor_d.upload(cos_gabor);

    if (cos_gabor.rows*cos_gabor.cols<=16*16){
        // cv::gpu::filter2D is limited to 16*16 kernels
        cv::gpu::filter2D(image_f_d, sin_response_d, -1, sin_gabor, cv::Point(-1,-1));
        cv::gpu::filter2D(image_f_d, cos_response_d, -1, cos_gabor, cv::Point(-1,-1));
    }
    else{
        int vertical_pad = (cos_gabor.rows-1)/2;
        int horizontal_pad = (cos_gabor.cols-1)/2;
        cv::gpu::copyMakeBorder(image_f_d,padded_image_d,
                                vertical_pad,vertical_pad,
                                horizontal_pad,horizontal_pad,
                                cv::BORDER_DEFAULT, 0.0);
	padded_image_d.download(temp);
	cv::imshow("Padded Image", temp);
	std::cout<<"Padded image size: "<<padded_image_d.size()<<std::endl;

        cv::gpu::convolve(padded_image_d, sin_gabor_d, sin_response_d);
        cv::gpu::convolve(padded_image_d, cos_gabor_d, cos_response_d);
    }
    cv::gpu::multiply(sin_response_d, sin_response_d, sin_response_d);
    cv::gpu::multiply(cos_response_d, cos_response_d, cos_response_d);

    std::cout<<"response size: "<<sin_response_d.size()<<std::endl;
	*/
    // Display results
    return 0;
}
