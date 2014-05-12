#include <string.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <omp.h>
#include <vector>

#include "ReinhardLocal.h"
#include "opencl/reinhardLocal.h"
#if ENABLE_HALIDE
#include "halide/blur_cpu.h"
#include "halide/blur_gpu.h"
#endif

/* 
Combines different exposure images to create HDR, as proposed here:
http://www.ceng.metu.edu.tr/~akyuz/files/hdrgpu.pdf
*/

using namespace hdr;

ReinhardLocal::ReinhardLocal(float _key, float _sat, float _epsilon, float _phi) : Filter() {
	m_name = "ReinhardLocal";
	key = _key;
	sat = _sat;
	epsilon = _epsilon;
	phi = _phi;
}

bool ReinhardLocal::setupOpenCL(cl_context_properties context_prop[], const Params& params) {
	return false;
}

double ReinhardLocal::runCLKernels() {
	return 0.0;	
}

bool ReinhardLocal::runOpenCL(int input_texid, int output_texid) {
	return false;
}

bool ReinhardLocal::runOpenCL(Image input, Image output) {

	//some parameters
	/*float key = 0.36f;
	float sat = 1.6f;

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -Dimage_size=%lu\n", input.width*input.height);

	if (!initCL(params, ReinhardLocal_kernel, flags)) {
		return false;
	}

	cl_int err;
	cl_kernel k_computeLogAvgLum, k_globalTMO;
	cl_mem mem_input, mem_output, mem_logAvgLum, mem_L_white;

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

	mem_L_white = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating L_white memory", return false);

	err  = clSetKernelArg(k_computeLogAvgLum, 0, sizeof(cl_mem), &mem_input);
	err  = clSetKernelArg(k_computeLogAvgLum, 1, sizeof(cl_mem), &mem_logAvgLum);
	err  = clSetKernelArg(k_computeLogAvgLum, 2, sizeof(cl_mem), &mem_L_white);
	CHECK_ERROR_OCL(err, "setting computeLogAvgLum arguments", return false);

	err  = clSetKernelArg(k_globalTMO, 0, sizeof(cl_mem), &mem_input);
	err  = clSetKernelArg(k_globalTMO, 1, sizeof(cl_mem), &mem_output);
	err  = clSetKernelArg(k_globalTMO, 2, sizeof(cl_mem), &mem_logAvgLum);
	err  = clSetKernelArg(k_globalTMO, 3, sizeof(cl_mem), &mem_L_white);
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
//	float* L_white = (float*) calloc(num_wg, sizeof(float));
//	err = clEnqueueReadBuffer(m_queue, mem_L_white, CL_TRUE, 0, sizeof(float)*num_wg, L_white, 0, NULL, NULL );
//	CHECK_ERROR_OCL(err, "reading image memory", return false);

//	printf("%f\n", logAvgLum[0]+logAvgLum[1]+logAvgLum[2]+logAvgLum[3]+logAvgLum[4]+logAvgLum[5]);
//
//	float L_white_max = 0;
//	for (int i=0; i<6; i++) {
//		if (L_white_max < L_white[i]) L_white_max = L_white[i];
//	}
//
//	printf("%f\n", L_white_max);

	reportStatus("Finished OpenCL kernel");

	bool passed = verify(input, output);
	reportStatus(
		"Finished in %lf ms (verification %s)",
		runTime*1000, passed ? "passed" : "failed");


	//cleanup
	clReleaseMemObject(mem_input);
	clReleaseMemObject(mem_output);
	clReleaseMemObject(mem_L_white);
	clReleaseMemObject(mem_logAvgLum);
	clReleaseKernel(k_globalTMO);
	clReleaseKernel(k_computeLogAvgLum);
	releaseCL();*/
	return false;
}

bool ReinhardLocal::cleanupOpenCL() {
	return false;
}

bool ReinhardLocal::runReference(Image input, Image output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*NUM_CHANNELS);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");


	const int mipmap_levels = 8;

	float logAvgLum = 0.f;
	float Lwhite = 0.f;	//smallest luminance that'll be mapped to pure white

	float* lum = (float*) calloc(input.height*input.width, sizeof(float));
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float3 rgb = {getPixel(input, x, y, 0),
				getPixel(input, x, y, 1), getPixel(input, x, y, 2)};

			float cur_lum = getPixelLuminance(rgb);
			if (cur_lum > Lwhite) Lwhite = cur_lum;

			logAvgLum += log(cur_lum + 0.000001);
			lum[x + y*input.width] = cur_lum;
		}
	}
	logAvgLum = exp(logAvgLum/(input.width*input.height));


	float factor = key/logAvgLum;
	float scale[mipmap_levels-1];

	Image* mipmap_pyramid = (Image*) calloc(mipmap_levels, sizeof(Image));
	mipmap_pyramid[0] = input;
	for (int i=1; i<mipmap_levels; i++) {
		mipmap_pyramid[i] = image_mipmap(input, i);
		scale[i-1] = pow(2, i-1);
	}

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float local_logAvgLum = 0.f;
			for (int i=0; i<mipmap_levels-1; i++) {
				int centre_x = x/pow(2, i);
				int centre_y = y/pow(2, i);
				int surround_x = x/pow(2, i+1);
				int surround_y = y/pow(2, i+1);

				float3 centre_pixel = { getPixel(mipmap_pyramid[i], centre_x, centre_y, 0)/((float)PIXEL_RANGE),
										getPixel(mipmap_pyramid[i], centre_x, centre_y, 1)/((float)PIXEL_RANGE),
										getPixel(mipmap_pyramid[i], centre_x, centre_y, 2)/((float)PIXEL_RANGE)};
				float3 surround_pixel= {getPixel(mipmap_pyramid[i+1], surround_x, surround_y, 0)/((float)PIXEL_RANGE),
										getPixel(mipmap_pyramid[i+1], surround_x, surround_y, 1)/((float)PIXEL_RANGE),
										getPixel(mipmap_pyramid[i+1], surround_x, surround_y, 2)/((float)PIXEL_RANGE)};
				float centre_logAvgLum = getPixelLuminance(centre_pixel);
				float surround_logAvgLum = getPixelLuminance(surround_pixel);


				float logAvgLum_diff = centre_logAvgLum - surround_logAvgLum;
				logAvgLum_diff = logAvgLum_diff >= 0 ? logAvgLum_diff : -logAvgLum_diff;

				if (logAvgLum_diff/(pow(2.f, phi)*key/(scale[i]*scale[i]) + centre_logAvgLum) > epsilon) {
					local_logAvgLum = centre_logAvgLum;
					break;
				}
				else local_logAvgLum = surround_logAvgLum;			
			}

			float3 rgb, xyz;
			rgb.x = getPixel(input, x, y, 0);
			rgb.y = getPixel(input, x, y, 1);
			rgb.z = getPixel(input, x, y, 2);

			xyz = RGBtoXYZ(rgb);

			float L  = (key/logAvgLum) * xyz.y;
			float Ld = L /(1.0 + local_logAvgLum);

			rgb.x = (pow(rgb.x/xyz.y, sat) * Ld)*PIXEL_RANGE;
			rgb.y = (pow(rgb.y/xyz.y, sat) * Ld)*PIXEL_RANGE;
			rgb.z = (pow(rgb.z/xyz.y, sat) * Ld)*PIXEL_RANGE;

			//printf("%f, %f, %f\n", rgb.x, rgb.y, rgb.z);

			setPixel(output, x, y, 0, rgb.x);
			setPixel(output, x, y, 1, rgb.y);
			setPixel(output, x, y, 2, rgb.z);
		}
	}



	reportStatus("Finished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new uchar[output.width*output.height*4];
	memcpy(m_reference.data, output.data, output.width*output.height*4);

	return true;
}