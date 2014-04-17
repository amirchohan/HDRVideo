#include <string.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <omp.h>
#include <vector>

#include "Reinhard.h"
#include "opencl/reinhard.h"
#if ENABLE_HALIDE
#include "halide/blur_cpu.h"
#include "halide/blur_gpu.h"
#endif

/* 
Combines different exposure images to create HDR, as proposed here:
http://www.ceng.metu.edu.tr/~akyuz/files/hdrgpu.pdf
*/

using namespace hdr;

Reinhard::Reinhard() : Filter() {
	m_name = "Reinhard";
}

bool Reinhard::runHalideCPU(Image input, Image output, const Params& params) {
	return false;
}

bool Reinhard::runHalideGPU(Image input, Image output, const Params& params) {
	return false;
}

bool Reinhard::runOpenCL(Image input, Image output, const Params& params) {

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -Dimage_size=%lu\n", input.width*input.height);

	if (!initCL(params, reinhard_kernel, flags)) {
		return false;
	}

	cl_int err;
	cl_kernel k_computeLogAvgLum, k_globalTMO;
	cl_mem mem_input, mem_output, mem_logAvgLum, mem_Ywhite;

	//set up kernels
	k_computeLogAvgLum = clCreateKernel(m_program, "computeLogAvgLum", &err);
	CHECK_ERROR_OCL(err, "creating computeLogAvgLum kernel", return false);

	k_globalTMO = clCreateKernel(m_program, "global_TMO", &err);
	CHECK_ERROR_OCL(err, "creating global_TMO kernel", return false);

	//get info to set the global_reduc and local_reduc according to the GPU specs
	size_t preferred_wg_size;	//workgroup size should be a multiple of this
	err = clGetKernelWorkGroupInfo (k_computeLogAvgLum, m_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", return false);

	size_t max_cu;	//max compute units
	err = clGetDeviceInfo(m_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &max_cu, NULL);

	const size_t local_reduc = preferred_wg_size;	//workgroup size for reduction kernels
	const size_t global_reduc = ((int)preferred_wg_size)*((int)max_cu);	//global_reduc size for reduction kernels
	const int num_wg = global_reduc/local_reduc;

	const size_t local = preferred_wg_size;	//workgroup size for normal kernels
	const size_t global = ceil((float)input.width*input.height/(float)local) * local;

	//memory objects
	mem_input = clCreateBuffer(m_clContext, CL_MEM_READ_ONLY, 
		sizeof(float)*input.width*input.height*4, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

	mem_output = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, 
		sizeof(float)*output.width*output.height*4, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

	mem_logAvgLum = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logAvgLum memory", return false);

	mem_Ywhite = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating Ywhite memory", return false);

	err  = clSetKernelArg(k_computeLogAvgLum, 0, sizeof(cl_mem), &mem_input);
	err  = clSetKernelArg(k_computeLogAvgLum, 1, sizeof(cl_mem), &mem_logAvgLum);
	err  = clSetKernelArg(k_computeLogAvgLum, 2, sizeof(cl_mem), &mem_Ywhite);
	CHECK_ERROR_OCL(err, "setting computeLogAvgLum arguments", return false);

	err  = clSetKernelArg(k_globalTMO, 0, sizeof(cl_mem), &mem_input);
	err  = clSetKernelArg(k_globalTMO, 1, sizeof(cl_mem), &mem_output);
	err  = clSetKernelArg(k_globalTMO, 2, sizeof(cl_mem), &mem_logAvgLum);
	err  = clSetKernelArg(k_globalTMO, 3, sizeof(cl_mem), &mem_Ywhite);
	CHECK_ERROR_OCL(err, "setting globalTMO arguments", return false);

	err = clEnqueueWriteBuffer(m_queue, mem_input, CL_TRUE, 0, 
		sizeof(float)*input.width*input.height*4, input.data, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "writing image memory", return false);



	//let it begin
	double start = omp_get_wtime();

	err = clEnqueueNDRangeKernel(m_queue, k_computeLogAvgLum, 1, NULL, &global_reduc, &local_reduc, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing computeLogAvgLum kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, k_globalTMO, 1, NULL, &global, &local, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing globalTMO kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);
	double runTime = omp_get_wtime() - start;


	//read results back
	err = clEnqueueReadBuffer(m_queue, mem_output, CL_TRUE, 0,
		sizeof(float)*output.width*output.height*4, output.data, 0, NULL, NULL );
	CHECK_ERROR_OCL(err, "reading image memory", return false);

//	float* logAvgLum = (float*) calloc(num_wg, sizeof(float));
//	err = clEnqueueReadBuffer(m_queue, mem_logAvgLum, CL_TRUE, 0, sizeof(float)*num_wg, logAvgLum, 0, NULL, NULL );
//	CHECK_ERROR_OCL(err, "reading image memory", return false);
//
//	float* Ywhite = (float*) calloc(num_wg, sizeof(float));
//	err = clEnqueueReadBuffer(m_queue, mem_Ywhite, CL_TRUE, 0, sizeof(float)*num_wg, Ywhite, 0, NULL, NULL );
//	CHECK_ERROR_OCL(err, "reading image memory", return false);

//	printf("%f\n", logAvgLum[0]+logAvgLum[1]+logAvgLum[2]+logAvgLum[3]+logAvgLum[4]+logAvgLum[5]);
//
//	float Ywhite_max = 0;
//	for (int i=0; i<6; i++) {
//		if (Ywhite_max < Ywhite[i]) Ywhite_max = Ywhite[i];
//	}
//
//	printf("%f\n", Ywhite_max);

	reportStatus("Finished OpenCL kernel");

	bool passed = verify(input, output);
	reportStatus(
		"Finished in %lf ms (verification %s)",
		runTime*1000, passed ? "passed" : "failed");


	//cleanup
	clReleaseMemObject(mem_input);
	clReleaseMemObject(mem_output);
	clReleaseMemObject(mem_Ywhite);
	clReleaseMemObject(mem_logAvgLum);
	clReleaseKernel(k_globalTMO);
	clReleaseKernel(k_computeLogAvgLum);
	releaseCL();
	return passed;
}

//computes gradient of a channel using forward difference
float* gradient_forward(float* input, int width, int height) {
	float* result = (float*) calloc(width*height, sizeof(float));

	for (int y = 0; y < height-1; y++) {
		for (int x = 0; x < width-1; x++) {
			result[x + y*width] = sqrt(pow(input[x+1 + y*width]-input[x + y*width], 2) + pow(input[x + (y+1)*width]-input[x + y*width], 2));
		}
		int x = width-1;
		result[x + y*width] = abs(input[x + (y+1)*width]-input[x + y*width]);
	}
	int y = height-1;
	for (int x = 0; x < width-1; x++) {
		result[x + y*width] = abs(input[x+1 + y*width]-input[x + y*width]);
	}
	return result;
}


//mipmaps the given channel input in 3x3 order
float* mipmap(float* input, int width, int height) {

	int m_width = width/2;
	int m_height = height/2;
	float* result = (float*) calloc(width*height, sizeof(float));

	for (int y = 0; y < m_height; y++) {
		for (int x = 0; x < m_width; x++) {
			int _x = 2*x;
			int _y = 2*y;
			result[x + y*m_width] = (input[_x + _y*width] + input[_x+1 + _y*width] + input[_x + (_y+1)*width] + input[(_x+1) + (_y+1)*width])/4.f;
		}
	}

	return result;
}


float* attenuate_func(float* lum, int width, int height) {
	float beta = 0.85;
	int k = 0;
	int k_width = width;	//width and height of the level k in the pyramid
	int k_height = height;
	float* k_lum = lum;
	float k_av_grad = 0.f;
	float* k_gradient;

	std::vector<float*> pyramid;
	std::vector<float> av_grads;
	std::vector<std::pair<unsigned int, unsigned int> > pyramid_sizes;

	for ( ; k_width >= 32 && k_height >= 32; k_height/=2, k_width/=2, k++) {

		//computing gradient magnitude using central differences at level k
		k_av_grad = 0.f;
		k_gradient = (float*) calloc(k_width*k_height, sizeof(float));
		for (int y = 0; y < k_height; y++) {
			for (int x = 0; x < k_width; x++) {
				int x_west = (x > 0) ? x-1 : 0;
				int x_east = (x < k_width-1) ? x+1 : k_width-1;
				int y_north = (y > 0) ? y-1 : 0;
				int y_south = (y < k_height-1) ? y+1 : k_height-1;

				float x_grad = (k_lum[x_west + y*k_width] - k_lum[x_east + y*k_width])/(float)pow(2.f, k+1);
				float y_grad = (k_lum[x + y_south*k_width] - k_lum[x + y_north*k_width])/(float)pow(2.f, k+1);
				k_gradient[x + y*k_width] = sqrt(pow(x_grad, 2) + pow(y_grad, 2));
				k_av_grad += k_gradient[x + y*k_width];
			}
		}
		pyramid.push_back(k_gradient);
		pyramid_sizes.push_back(std::pair< unsigned int, unsigned int >(k_width, k_height));
		av_grads.push_back(k_av_grad/(k_width*k_height));

		k_lum = mipmap(k_lum, k_width, k_height);
	}


	//computing attenuation functions
	k_gradient = pyramid.back();
	k_width = pyramid_sizes.back().first;
	k_height = pyramid_sizes.back().second;
	float k_alpha = 0.1*av_grads.back();
	float* k_atten_func;
	k--;

	//attenuation function for the coarsest level
	k_atten_func = (float*) calloc(k_width*k_height, sizeof(float));
	for (int y = 0; y < k_height; y++) {
		for (int x = 0; x < k_width; x++) {
			k_atten_func[x + y*k_width] = (k_alpha/k_gradient[x + y*k_width])*pow(k_gradient[x + y*k_width]/k_alpha, beta);
		}
	}

	pyramid.pop_back();
	pyramid_sizes.pop_back();
	av_grads.pop_back();

	while (! pyramid.empty()) {
		
		float* k1_atten_func = k_atten_func;
		int k1_width = k_width;
		int k1_height = k_height;

		k_gradient = pyramid.back();
		k_width = pyramid_sizes.back().first;
		k_height = pyramid_sizes.back().second;
		float k_alpha = 0.1*av_grads.back();
		float k_xy_scale_factor;
		float k_xy_atten_func;
		k--;

		//attenuation function for this level
		k_atten_func = (float*) calloc(k_width*k_height, sizeof(float));
		for (int y = 0; y < k_height; y++) {
			for (int x = 0; x < k_width; x++) {

				int k1_x = x/2, k1_y = y/2;		//x and y value of the coarser grid

				//neighbours need to be left or right dependent on where we are
				int n_x = (x & 1) ? 1 : -1;
				int n_y = (y & 1) ? 1 : -1;


				//this stops us from going out of bounds
				if ((k1_x + n_x) < 0) n_x = 0;
				if ((k1_y + n_y) < 0) n_y = 0;
				if ((k1_x + n_x) >= k1_width) n_x = 0;
				if ((k1_y + n_y) >= k1_height) n_y = 0;
				if (k1_x == k1_width) k1_x -= 1;
				if (k1_y == k1_height) k1_y -= 1;

				k_xy_atten_func = 9.0*k1_atten_func[k1_x 		+ k1_y			*k1_width]
								+ 3.0*k1_atten_func[k1_x+n_x 	+ k1_y			*k1_width]
								+ 3.0*k1_atten_func[k1_x 		+ (k1_y+n_y)	*k1_width]
								+ 1.0*k1_atten_func[k1_x+n_x 	+ (k1_y+n_y)	*k1_width];

				k_xy_scale_factor = (k_alpha/k_gradient[x + y*k_width])*pow(k_gradient[x + y*k_width]/k_alpha, beta);
				k_atten_func[x + y*k_width] = (1.f/16.f)*(k_xy_atten_func)*k_xy_scale_factor;
			}
		}

		pyramid.pop_back();
		pyramid_sizes.pop_back();
		av_grads.pop_back();
	}
	return k_atten_func;
}

bool Reinhard::runReference(Image input, Image output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	//computing logarithmic luminace of the image
	float* lum = (float*) calloc(input.width * input.height, sizeof(float));	//logarithm luminance
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			float3 org_pixel = {getPixel(input, x, y, 0), getPixel(input, x, y, 1), getPixel(input, x, y, 2)};
			lum[x + y*input.width] = getPixelLuminance(org_pixel) + 0.000001;
		}
	}

	//computing the attenuation function which will then be multiplied by luminance gradient to acheive attenuated gradient
	float* att_func = attenuate_func(lum, input.width, input.height);	//o(x,y)

	//luminance gradient in forward direction for x and y
	float* grad_x = (float*) calloc(input.width * input.height, sizeof(float));	//H(x,y)
	float* grad_y = (float*) calloc(input.width * input.height, sizeof(float));	//H(x,y)
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			grad_x[x + y*input.width] = (x < input.width)  ? (lum[x+1 +    y*input.width] - lum[x + y*input.width]) : 0;
			grad_y[x + y*input.width] = (y < input.height) ? (lum[x   + (y+1)*input.width] -lum[x + y*input.width]) : 0;
		}
	}

	//attenuated gradient achieved by using the previously computed attenuation function
	float* att_grad_x = (float*) calloc(input.height * input.width, sizeof(float));	//G(x,y)
	float* att_grad_y = (float*) calloc(input.height * input.width, sizeof(float));	//G(x,y)
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			att_grad_x[x + y*input.width] = grad_x[x + y*input.width] * att_func[x + y*input.width];
			att_grad_y[x + y*input.width] = grad_y[x + y*input.width] * att_func[x + y*input.width];
		}
	}

	//divG(x,y)
	float* div_grad = (float*) calloc(input.height * input.width, sizeof(float));
	div_grad[0] = 0;
	for (int x = 1; x < input.width; x++) {
		div_grad[x] = grad_x[x] - grad_x[x-1];
	}
	for (int y = 1; y < input.height; y++) {
		div_grad[y*input.width] = grad_y[y*input.width] - grad_y[(y-1)*input.width];
		for (int x = 1; x < input.width; x++) {
			div_grad[x + y*input.width] = (grad_x[x + y*input.width] - grad_x[(x-1) + y*input.width]) + (grad_y[x + y*input.width] - grad_y[x + (y-1)*input.width]);
		}
	}


	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			setPixel(output, x, y, 0, div_grad[x + y*input.width]);
			setPixel(output, x, y, 1, div_grad[x + y*input.width]);
			setPixel(output, x, y, 2, div_grad[x + y*input.width]);
		}
	}

	reportStatus("Finished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*4];
	memcpy(m_reference.data, output.data, output.width*output.height*4);

	return true;
}

/*
bool Reinhard::runReference(Image input, Image output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	float logAvgLum = 0;
	float Ywhite = 0.f;

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float3 hdr = {getPixel(input, x, y, 0),
				getPixel(input, x, y, 1), getPixel(input, x, y, 2)};

			float lum = getPixelLuminance(hdr);
			if (lum > Ywhite) Ywhite = lum;

			logAvgLum += log(lum + 0.000001);
		}
	}

	float key = 1.0f;
	float sat = 1.5f;

	logAvgLum = exp(logAvgLum/(input.width*input.height));

	//global_reduc Tone-mapping operator
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			float3 rgb, xyz;
			rgb.x = getPixel(input, x, y, 0);
			rgb.y = getPixel(input, x, y, 1);
			rgb.z = getPixel(input, x, y, 2);

			xyz = RGBtoXYZ(rgb);

			float Y  = (key/logAvgLum) * xyz.y;
			float Yd = (Y * (1.0 + Y/(Ywhite * Ywhite)) )/(1.0 + Y);

			rgb.x = pow(rgb.x/xyz.y, sat) * Yd;
			rgb.y = pow(rgb.y/xyz.y, sat) * Yd;
			rgb.z = pow(rgb.z/xyz.y, sat) * Yd;

			setPixel(output, x, y, 0, rgb.x);
			setPixel(output, x, y, 1, rgb.y);
			setPixel(output, x, y, 2, rgb.z);
		}
	}

	/*float factor = key/logAvgLum;
	float epsilon = 0.05;
	float phi = 8.0;
	float scale[7] = {1, 2, 4, 8, 16, 32, 64};
	float v1, v2;


	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			float La = 0;

			for (int level=1; level<6; level++) {

				int sc = scale[level]/2;
				float r1 = 0, r2 = 0;
				float v1 = 0, v2 = 0;


				for (int j = -sc; j <= sc; j++) {
					for (int i = -sc; i <= sc; i++) {
						int _x = clamp(x+i, 0, input.width-1);
						int _y = clamp(y+j, 0, input.height-1);

						r1 = ( 1/(phi*pow(level*scale[level], 2)) ) *
							exp(- (pow(_x,2) + pow(_y,2))/pow(level*scale[level], 2) );
						v1 += hdr_luminance[_x + _y*input.width]*r1;


						r2 = (1/(phi*pow((level+1)*scale[level+1], 2)) ) *
							exp(- (pow(_x,2) + pow(_y,2))/pow((level+1)*scale[level+1], 2));
						v2 += hdr_luminance[_x + _y*input.width]*r2;

					}
				}
				v1 *= factor;
				v2 *= factor;

				float V = abs(v1 - v2) / (pow(2, phi) * key/(scale[level] * scale[level]) + v1);

				if (V > epsilon) {
					La = v1;
					break;
				}
				else La = v2;
			}

			//printf("%f\n", La);

			float3 rgb, xyz;
			rgb.x = getPixel(input, x, y, 0)/255.f;
			rgb.y = getPixel(input, x, y, 1)/255.f;
			rgb.z = getPixel(input, x, y, 2)/255.f;

			xyz = RGBtoXYZ(rgb);

			float Y  = (key/logAvgLum) * xyz.y;
			float Yd = Y /(1.0 + La);

			rgb.x = pow(rgb.x/xyz.x, sat) * Yd;
			rgb.y = pow(rgb.y/xyz.y, sat) * Yd;
			rgb.z = pow(rgb.z/xyz.z, sat) * Yd;

			//printf("%f, %f, %f\n", rgb.x, rgb.y, rgb.z);

			setPixel(output, x, y, 0, rgb.x*255.f);
			setPixel(output, x, y, 1, rgb.y*255.f);
			setPixel(output, x, y, 2, rgb.z*255.f);
		}
		printf("%d\n", y);
	}



	reportStatus("Finished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*4];
	memcpy(m_reference.data, output.data, output.width*output.height*4);

	return true;
}*/