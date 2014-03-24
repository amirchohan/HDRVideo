#include <string.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <omp.h>

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
	m_type = TONEMAP;
}

bool Reinhard::runHalideCPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool Reinhard::runHalideGPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool Reinhard::runOpenCL(LDRI input, Image output, const Params& params) {

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
	mem_input = clCreateBuffer(m_context, CL_MEM_READ_ONLY, 
		sizeof(float)*input.width*input.height*4, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

	mem_output = clCreateBuffer(m_context, CL_MEM_READ_WRITE, 
		sizeof(float)*output.width*output.height*4, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

	mem_logAvgLum = clCreateBuffer(m_context, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logAvgLum memory", return false);

	mem_Ywhite = clCreateBuffer(m_context, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
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
		sizeof(float)*input.width*input.height*4, input.images[0].data, 0, NULL, NULL);
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


bool Reinhard::runReference(LDRI input, Image output) {

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

			float3 hdr = {getPixel(input.images[0], x, y, 0),
				getPixel(input.images[0], x, y, 1), getPixel(input.images[0], x, y, 2)};

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
			rgb.x = getPixel(input.images[0], x, y, 0);
			rgb.y = getPixel(input.images[0], x, y, 1);
			rgb.z = getPixel(input.images[0], x, y, 2);

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
	}*/



	reportStatus("Finished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*4];
	memcpy(m_reference.data, output.data, output.width*output.height*4);

	return true;
}
