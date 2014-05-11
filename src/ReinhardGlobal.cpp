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

bool ReinhardGlobal::setupOpenCL(cl_context_properties context_prop[], const Params& params) {

	//some parameters
	float key = 0.18f;
	float sat = 1.6f;

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -D NUM_CHANNELS=%d -Dimage_size=%d -D WIDTH=%d -D HEIGHT=%d",
				NUM_CHANNELS, image_width*image_height, image_width, image_height);

	if (!initCL(context_prop, params, reinhardGlobal_kernel, flags)) {
		return false;
	}

	cl_int err;

	/////////////////////////////////////////////////////////////////kernels

	kernels["transfer_data"] = clCreateKernel(m_program, "transfer_data", &err);
	CHECK_ERROR_OCL(err, "creating transfer_data kernel", return false);

	//this kernel computes log average luminance of the image
	kernels["computeLogAvgLum"] = clCreateKernel(m_program, "computeLogAvgLum", &err);
	CHECK_ERROR_OCL(err, "creating computeLogAvgLum kernel", return false);

	//performs the reinhard global tone mapping operator
	kernels["global_TMO"] = clCreateKernel(m_program, "global_TMO", &err);
	CHECK_ERROR_OCL(err, "creating global_TMO kernel", return false);

	kernels["transfer_data_output"] = clCreateKernel(m_program, "transfer_data_output", &err);
	CHECK_ERROR_OCL(err, "creating transfer_data kernel", return false);

	/////////////////////////////////////////////////////////////////kernel sizes

	size_t max_cu;	//max compute units
	err = clGetDeviceInfo(m_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(size_t), &max_cu, NULL);

	//get info to set the global_reduc and local_reduc according to the GPU specs
	size_t preferred_wg_size;	//workgroup size should be a multiple of this
	err = clGetKernelWorkGroupInfo (kernels["computeLogAvgLum"], m_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", return false);

	size_t local = preferred_wg_size;	//workgroup size for reduction kernels
	size_t global = ((int)preferred_wg_size)*((int)max_cu);	//global_reduc size for reduction kernels
	const int num_wg = global/local;

	global_sizes["reduc"] = global;
	local_sizes["reduc"] = local;

	err = clGetKernelWorkGroupInfo (kernels["global_TMO"], m_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", return false);

	local = preferred_wg_size;	//workgroup size for normal kernels
	global = ceil((float)image_width*image_height/(float)local) * local;

	global_sizes["normal"] = global;
	local_sizes["normal"] = local;

	reportStatus("---------------------------------Kernel transfer_data:");

	size_t max_wg_size;	//max workgroup size for the kernel
	err = clGetKernelWorkGroupInfo (kernels["transfer_data"], m_device,
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_WORK_GROUP_SIZE", return false);
	reportStatus("CL_KERNEL_WORK_GROUP_SIZE: %lu", max_wg_size);

	err = clGetKernelWorkGroupInfo (kernels["transfer_data"], m_device,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_size, NULL);
	CHECK_ERROR_OCL(err, "getting CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE", return false);
	reportStatus("CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: %lu", preferred_wg_size);

	size_t* local2Dsize = (size_t*) calloc(2, sizeof(size_t));
	size_t* global2Dsize = (size_t*) calloc(2, sizeof(size_t));
	local2Dsize[0] = preferred_wg_size;
	local2Dsize[1] = preferred_wg_size;
	while (local2Dsize[0]*local2Dsize[1] > max_wg_size) {
		local2Dsize[0] /= 2;
		local2Dsize[1] /= 2;
	}
	global2Dsize[0] = image_width;
	global2Dsize[1] = image_height;

	global2Dsize[0] = ceil((float)global2Dsize[0]/(float)local2Dsize[0])*local2Dsize[0];
	global2Dsize[1] = ceil((float)global2Dsize[1]/(float)local2Dsize[1])*local2Dsize[1];

	twoDlocal_sizes["transfer_data"] = local2Dsize;
	twoDglobal_sizes["transfer_data"] = global2Dsize;

	reportStatus("Kernel sizes: Local=(%lu, %lu) Global=(%lu, %lu)", local2Dsize[0], local2Dsize[1], global2Dsize[0], global2Dsize[1]);

	twoDlocal_sizes["transfer_data_output"] = local2Dsize;
	twoDglobal_sizes["transfer_data_output"] = global2Dsize;


	/////////////////////////////////////////////////////////////////allocating memory

	mems["image"] = clCreateBuffer(m_clContext, CL_MEM_READ_ONLY, sizeof(uchar)*image_width*image_height*NUM_CHANNELS, NULL, &err);
	CHECK_ERROR_OCL(err, "creating image memory", return false);

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

	err  = clSetKernelArg(kernels["transfer_data"], 0, sizeof(cl_mem), &mem_images[0]);
	err |= clSetKernelArg(kernels["transfer_data"], 1, sizeof(cl_mem), &mems["image"]);
	CHECK_ERROR_OCL(err, "setting transfer_data arguments", return false);

	err  = clSetKernelArg(kernels["computeLogAvgLum"], 0, sizeof(cl_mem), &mems["image"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 1, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 2, sizeof(cl_mem), &mems["Lwhite"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 3, sizeof(float*)*local_sizes["reduc"], NULL);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 4, sizeof(float*)*local_sizes["reduc"], NULL);
	CHECK_ERROR_OCL(err, "setting computeLogAvgLum arguments", return false);

	err  = clSetKernelArg(kernels["global_TMO"], 0, sizeof(cl_mem), &mems["image"]);
	err  = clSetKernelArg(kernels["global_TMO"], 1, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["global_TMO"], 2, sizeof(cl_mem), &mems["Lwhite"]);
	err  = clSetKernelArg(kernels["global_TMO"], 3, sizeof(float), &key);
	err  = clSetKernelArg(kernels["global_TMO"], 4, sizeof(float), &sat);
	err  = clSetKernelArg(kernels["global_TMO"], 5, sizeof(unsigned int), &num_wg);
	CHECK_ERROR_OCL(err, "setting globalTMO arguments", return false);

	err  = clSetKernelArg(kernels["transfer_data_output"], 0, sizeof(cl_mem), &mem_images[1]);
	err |= clSetKernelArg(kernels["transfer_data_output"], 1, sizeof(cl_mem), &mems["image"]);
	CHECK_ERROR_OCL(err, "setting transfer_data_output arguments", return false);

	return true;
}

double ReinhardGlobal::runCLKernels() {
	double start = omp_get_wtime();

	cl_int err;
	err = clEnqueueNDRangeKernel(m_queue, kernels["transfer_data"], 2, NULL, twoDglobal_sizes["transfer_data"], twoDlocal_sizes["transfer_data"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing transfer_data kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["computeLogAvgLum"], 1, NULL, &global_sizes["reduc"], &local_sizes["reduc"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing computeLogAvgLum kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["global_TMO"], 1, NULL, &global_sizes["normal"], &local_sizes["normal"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing globalTMO kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["transfer_data_output"], 2, NULL, twoDglobal_sizes["transfer_data_output"], twoDlocal_sizes["transfer_data_output"], 0, NULL, NULL);
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