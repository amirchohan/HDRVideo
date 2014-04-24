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

ReinhardGlobal::ReinhardGlobal() : Filter() {
	m_name = "ReinhardGlobal";
}

bool ReinhardGlobal::runHalideCPU(Image input, Image output, const Params& params) {
	return false;
}

bool ReinhardGlobal::runHalideGPU(Image input, Image output, const Params& params) {
	return false;
}

bool ReinhardGlobal::setupOpenCL(cl_context_properties context_prop[], const Params& params, const int image_size) {

	//some parameters
	float key = 0.18f;
	float sat = 1.6f;

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -D NUM_CHANNELS=%d -Dimage_size=%d", NUM_CHANNELS, image_size);

	if (!initCL(context_prop, params, reinhardGlobal_kernel, flags)) {
		return false;
	}

	cl_int err;

	/////////////////////////////////////////////////////////////////kernels
	//this kernel computes log average luminance of the image
	kernels["computeLogAvgLum"] = clCreateKernel(m_program, "computeLogAvgLum", &err);
	CHECK_ERROR_OCL(err, "creating computeLogAvgLum kernel", return false);

	//performs the reinhard global tone mapping operator
	kernels["global_TMO"] = clCreateKernel(m_program, "global_TMO", &err);
	CHECK_ERROR_OCL(err, "creating global_TMO kernel", return false);

	//get info to set the global_reduc and local_reduc according to the GPU specs
	size_t preferred_wg_size;	//workgroup size should be a multiple of this
	err = clGetKernelWorkGroupInfo (kernels["computeLogAvgLum"], m_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", return false);

	size_t max_cu;	//max compute units
	err = clGetDeviceInfo(m_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &max_cu, NULL);

	const size_t local_reduc = preferred_wg_size;	//workgroup size for reduction kernels
	const size_t global_reduc = ((int)preferred_wg_size)*((int)max_cu);	//global_reduc size for reduction kernels
	const int num_wg = global_reduc/local_reduc;

	const size_t local = preferred_wg_size;	//workgroup size for normal kernels
	const size_t global = ceil((float)image_size/(float)local) * local;

	global_sizes["reduc"] = global_reduc;
	global_sizes["normal"] = global;

	local_sizes["reduc"] = local_reduc;
	local_sizes["normal"] = local;


	/////////////////////////////////////////////////////////////////allocating memory

	mems["input"] = clCreateBuffer(m_clContext, CL_MEM_READ_ONLY, sizeof(float)*image_size*NUM_CHANNELS, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

	mems["output"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*image_size*NUM_CHANNELS, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

	mems["logAvgLum"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logAvgLum memory", return false);

	mems["Lwhite"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating Lwhite memory", return false);


	/////////////////////////////////////////////////////////////////setting kernel arguements

	err  = clSetKernelArg(kernels["computeLogAvgLum"], 0, sizeof(cl_mem), &mems["input"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 1, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 2, sizeof(cl_mem), &mems["Lwhite"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 3, sizeof(float*)*local_reduc, NULL);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 4, sizeof(float*)*local_reduc, NULL);
	CHECK_ERROR_OCL(err, "setting computeLogAvgLum arguments", return false);

	err  = clSetKernelArg(kernels["global_TMO"], 0, sizeof(cl_mem), &mems["input"]);
	err  = clSetKernelArg(kernels["global_TMO"], 1, sizeof(cl_mem), &mems["output"]);
	err  = clSetKernelArg(kernels["global_TMO"], 2, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["global_TMO"], 3, sizeof(cl_mem), &mems["Lwhite"]);
	err  = clSetKernelArg(kernels["global_TMO"], 4, sizeof(float), &key);
	err  = clSetKernelArg(kernels["global_TMO"], 5, sizeof(float), &sat);
	err  = clSetKernelArg(kernels["global_TMO"], 6, sizeof(unsigned int), &num_wg);
	CHECK_ERROR_OCL(err, "setting globalTMO arguments", return false);

	return true;
}

double ReinhardGlobal::runCLKernels() {
	double start = omp_get_wtime();

	cl_int err;
	err = clEnqueueNDRangeKernel(m_queue, kernels["computeLogAvgLum"], 1, NULL, &global_sizes["reduc"], &local_sizes["reduc"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing computeLogAvgLum kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["global_TMO"], 1, NULL, &global_sizes["normal"], &local_sizes["normal"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing globalTMO kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);
	return omp_get_wtime() - start;	
}


bool ReinhardGlobal::runOpenCL(int gl_texture) {
	//GLuint textures[1];
	//glGenTextures(1, &textures[0]);
	//glBindTexture(GL_TEXTURE_2D, textures[0]);
	////int height;
	////glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, height);
//
//	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, input.width, input.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
//
//	//cl_mem hello = clCreateFromGLTexture2D(m_clContext, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, textures[0], &err);
	//CHECK_ERROR_OCL(err, "creating from GL texture", return false);


	//glDeleteTextures(1, &textures[0]);

	return false;
}


//when image data is provided in form of Image data structure as opposed to an OpenGL texture
bool ReinhardGlobal::runOpenCL(Image input, Image output) {

	cl_int err;

	//transfer memory to the device
	err = clEnqueueWriteBuffer(m_queue, mems["input"], CL_TRUE, 0, sizeof(float)*input.width*input.height*NUM_CHANNELS, input.data, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "writing image memory", return false);

	//let it begin
	double runTime = runCLKernels();

	//read results back
	err = clEnqueueReadBuffer(m_queue, mems["output"], CL_TRUE, 0,	sizeof(float)*output.width*output.height*NUM_CHANNELS, output.data, 0, NULL, NULL );
	CHECK_ERROR_OCL(err, "reading image memory", return false);

	reportStatus("Finished OpenCL kernel");

	bool passed = verify(input, output);
	reportStatus(
		"Finished in %lf ms (verification %s)",
		runTime*1000, passed ? "passed" : "failed");

	return passed;
}


bool ReinhardGlobal::cleanupOpenCL() {
	clReleaseMemObject(mems["input"]);
	clReleaseMemObject(mems["output"]);
	clReleaseMemObject(mems["Lwhite"]);
	clReleaseMemObject(mems["logAvgLum"]);
	clReleaseKernel(kernels["global_TMO"]);
	clReleaseKernel(kernels["computeLogAvgLum"]);
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

	//some parameters
	float key = 0.18f;
	float sat = 1.6f;

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

			setPixel(output, x, y, 0, rgb.x);
			setPixel(output, x, y, 1, rgb.y);
			setPixel(output, x, y, 2, rgb.z);
		}
	}

	reportStatus("Finished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*NUM_CHANNELS];
	memcpy(m_reference.data, output.data, output.width*output.height*NUM_CHANNELS);

	return true;
}