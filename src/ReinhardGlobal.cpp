#include <string.h>
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <omp.h>

//#include <GLES/gl.h>

#include "ReinhardGlobal.h"
#include "opencl/reinhardGlobal.h"
#if ENABLE_HALIDE
#include "halide/blur_cpu.h"
#include "halide/blur_gpu.h"
#endif

/* 
Combines different exposure images to create HDR, as proposed here:
http://www.ceng.metu.edu.tr/~akyuz/files/hdrgpu.pdf
*/

using namespace hdr;

ReinhardGlobal::ReinhardGlobal(float _key, float _sat) : Filter() {
	m_name = "ReinhardGlobal";
	key = _key;
	sat = _sat;
}

bool ReinhardGlobal::setupOpenCL(cl_context_properties context_prop[], const Params& params) {

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -D NUM_CHANNELS=%d -Dimage_size=%d -D WIDTH=%d -D HEIGHT=%d",
				NUM_CHANNELS, image_width*image_height, image_width, image_height);

	if (!initCL(context_prop, params, reinhardGlobal_kernel, flags)) {
		return false;
	}

	cl_int err;

	/////////////////////////////////////////////////////////////////kernels

	//this kernel computes log average luminance of the image
	kernels["computeLogAvgLum"] = clCreateKernel(m_program, "computeLogAvgLum", &err);
	CHECK_ERROR_OCL(err, "creating computeLogAvgLum kernel", return false);

	//this kernel computes log average luminance of the image
	kernels["finalReduc"] = clCreateKernel(m_program, "finalReduc", &err);
	CHECK_ERROR_OCL(err, "creating finalReduc kernel", return false);

	//performs the reinhard global tone mapping operator
	kernels["reinhardGlobal"] = clCreateKernel(m_program, "reinhardGlobal", &err);
	CHECK_ERROR_OCL(err, "creating reinhardGlobal kernel", return false);


	/////////////////////////////////////////////////////////////////kernel sizes

	kernel2DSizes("computeLogAvgLum");
	kernel2DSizes("reinhardGlobal");

	reportStatus("---------------------------------Kernel finalReduc:");

		int num_wg = (global_sizes["computeLogAvgLum"][0]*global_sizes["computeLogAvgLum"][1])
						/(local_sizes["computeLogAvgLum"][0]*local_sizes["computeLogAvgLum"][1]);
		reportStatus("Number of work groups in computeLogAvgLum: %lu", num_wg);
	
		size_t* local = (size_t*) calloc(2, sizeof(size_t));
		size_t* global = (size_t*) calloc(2, sizeof(size_t));
		local[0] = num_wg;	//workgroup size for normal kernels
		global[0] = num_wg;
	
		local_sizes["finalReduc"] = local;
		global_sizes["finalReduc"] = global;
		reportStatus("Kernel sizes: Local=%lu Global=%lu", local[0], global[0]);


	/////////////////////////////////////////////////////////////////allocating memory

	mems["logAvgLum"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logAvgLum memory", return false);

	mems["Lwhite"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating Lwhite memory", return false);

	if (params.opengl) {
		mem_images[0] = clCreateFromGLTexture2D(m_clContext, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, in_tex, &err);
		CHECK_ERROR_OCL(err, "creating gl input texture", return false);
		
		mem_images[1] = clCreateFromGLTexture2D(m_clContext, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, out_tex, &err);
		CHECK_ERROR_OCL(err, "creating gl output texture", return false);
	}
	else {
		cl_image_format format;
		format.image_channel_order = CL_RGBA;
		format.image_channel_data_type = CL_UNSIGNED_INT8;
		mem_images[0] = clCreateImage2D(m_clContext, CL_MEM_READ_ONLY, &format, image_width, image_height, 0, NULL, &err);
		CHECK_ERROR_OCL(err, "creating input image memory", return false);
	
		mem_images[1] = clCreateImage2D(m_clContext, CL_MEM_WRITE_ONLY, &format, image_width, image_height, 0, NULL, &err);
		CHECK_ERROR_OCL(err, "creating output image memory", return false);
	}

	/////////////////////////////////////////////////////////////////setting kernel arguements

	err  = clSetKernelArg(kernels["computeLogAvgLum"], 0, sizeof(cl_mem), &mem_images[0]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 1, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 2, sizeof(cl_mem), &mems["Lwhite"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 3, sizeof(float*)*local_sizes["computeLogAvgLum"][0]*local_sizes["computeLogAvgLum"][1], NULL);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 4, sizeof(float*)*local_sizes["computeLogAvgLum"][0]*local_sizes["computeLogAvgLum"][1], NULL);
	CHECK_ERROR_OCL(err, "setting computeLogAvgLum arguments", return false);

	err  = clSetKernelArg(kernels["finalReduc"], 0, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["finalReduc"], 1, sizeof(cl_mem), &mems["Lwhite"]);
	err  = clSetKernelArg(kernels["finalReduc"], 2, sizeof(unsigned int), &num_wg);
	CHECK_ERROR_OCL(err, "setting finalReduc arguments", return false);

	err  = clSetKernelArg(kernels["reinhardGlobal"], 0, sizeof(cl_mem), &mem_images[0]);
	err  = clSetKernelArg(kernels["reinhardGlobal"], 1, sizeof(cl_mem), &mem_images[1]);
	err  = clSetKernelArg(kernels["reinhardGlobal"], 2, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["reinhardGlobal"], 3, sizeof(cl_mem), &mems["Lwhite"]);
	err  = clSetKernelArg(kernels["reinhardGlobal"], 4, sizeof(float), &key);
	err  = clSetKernelArg(kernels["reinhardGlobal"], 5, sizeof(float), &sat);
	CHECK_ERROR_OCL(err, "setting globalTMO arguments", return false);

	reportStatus("\n\n");

	return true;
}

double ReinhardGlobal::runCLKernels() {
	double start = omp_get_wtime();

	cl_int err;
	err = clEnqueueNDRangeKernel(m_queue, kernels["computeLogAvgLum"], 2, NULL, global_sizes["computeLogAvgLum"], local_sizes["computeLogAvgLum"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing computeLogAvgLum kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["finalReduc"], 1, NULL, &global_sizes["finalReduc"][0], &local_sizes["finalReduc"][0], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing finalReduc kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["reinhardGlobal"], 2, NULL, global_sizes["reinhardGlobal"], local_sizes["reinhardGlobal"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing transfer_data kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);
	return omp_get_wtime() - start;	
}


bool ReinhardGlobal::runOpenCL(int input_texid, int output_texid) {
	cl_int err;

	err = clEnqueueAcquireGLObjects(m_queue, 2, &mem_images[0], 0, 0, 0);
	CHECK_ERROR_OCL(err, "acquiring GL objects", return false);

	double runTime = runCLKernels();

	err = clEnqueueReleaseGLObjects(m_queue, 2, &mem_images[0], 0, 0, 0);
	CHECK_ERROR_OCL(err, "releasing GL objects", return false);

	reportStatus("Finished OpenCL kernels in %lf ms", runTime*1000);

	return false;
}

//when image data is provided in form of Image data structure as opposed to an OpenGL texture
bool ReinhardGlobal::runOpenCL(Image input, Image output) {

	cl_int err;

 	const size_t origin[] = {0, 0, 0};
 	const size_t region[] = {input.width, input.height, 1};
	err = clEnqueueWriteImage(m_queue, mem_images[0], CL_TRUE, origin, region, sizeof(uchar)*input.width*NUM_CHANNELS, 0, input.data, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "writing image memory", return false);

	//let it begin
	double runTime = runCLKernels();

	//read results back
	err = clEnqueueReadImage(m_queue, mem_images[1], CL_TRUE, origin, region, sizeof(uchar)*input.width*NUM_CHANNELS, 0, output.data, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "reading image memory", return false);

	reportStatus("Finished OpenCL kernel");

	bool passed = verify(input, output);
	reportStatus(
		"Finished in %lf ms (verification %s)",
		runTime*1000, passed ? "passed" : "failed");

	return passed;
}


bool ReinhardGlobal::cleanupOpenCL() {
	clReleaseMemObject(mem_images[0]);
	clReleaseMemObject(mem_images[1]);
	clReleaseMemObject(mems["Lwhite"]);
	clReleaseMemObject(mems["logAvgLum"]);
	clReleaseKernel(kernels["computeLogAvgLum"]);
	clReleaseKernel(kernels["finalReduc"]);
	clReleaseKernel(kernels["reinhardGlobal"]);
	releaseCL();
	return true;
}


bool ReinhardGlobal::runReference(Image input, Image output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	float logAvgLum = 0.f;
	float Lwhite = 0.f;	//smallest luminance that'll be mapped to pure white

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float3 hdr = {getPixel(input, x, y, 0),
				getPixel(input, x, y, 1), getPixel(input, x, y, 2)};

			float lum = getPixelLuminance(hdr);
			logAvgLum += log(lum + 0.000001);

			if (lum > Lwhite) Lwhite = lum;
		}
	}
	logAvgLum = exp(logAvgLum/(input.width*input.height));

	//Global Tone-mapping operator
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			float3 rgb, xyz;
			rgb.x = getPixel(input, x, y, 0);
			rgb.y = getPixel(input, x, y, 1);
			rgb.z = getPixel(input, x, y, 2);

			xyz = RGBtoXYZ(rgb);

			float L  = (key/logAvgLum) * xyz.y;
			float Ld = (L * (1.f + L/(Lwhite * Lwhite)) )/(1.f + L);

			rgb.x = pow(rgb.x/xyz.y, sat) * Ld;
			rgb.y = pow(rgb.y/xyz.y, sat) * Ld;
			rgb.z = pow(rgb.z/xyz.y, sat) * Ld;

			setPixel(output, x, y, 0, rgb.x*PIXEL_RANGE);
			setPixel(output, x, y, 1, rgb.y*PIXEL_RANGE);
			setPixel(output, x, y, 2, rgb.z*PIXEL_RANGE);
		}
	}

	reportStatus("Finished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new uchar[output.width*output.height*NUM_CHANNELS];
	memcpy(m_reference.data, output.data, output.width*output.height*NUM_CHANNELS);

	return true;
}