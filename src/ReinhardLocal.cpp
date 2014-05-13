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
	num_mipmaps = 8;
}

bool ReinhardLocal::setupOpenCL(cl_context_properties context_prop[], const Params& params) {

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -D NUM_CHANNELS=%d -Dimage_size=%d -D WIDTH=%d -D HEIGHT=%d -D NUM_MIPMAPS=%d -D PHI=%f -D EPSILON=%f",
				NUM_CHANNELS, image_width*image_height, image_width, image_height, num_mipmaps, phi, epsilon);

	if (!initCL(context_prop, params, reinhardLocal_kernel, flags)) {
		return false;
	}

	cl_int err;

	/////////////////////////////////////////////////////////////////kernels

	//this kernel computes log average luminance of the image
	kernels["computeLogAvgLum"] = clCreateKernel(m_program, "computeLogAvgLum", &err);
	CHECK_ERROR_OCL(err, "creating computeLogAvgLum kernel", return false);

	//this kernel computes log average luminance of the image
	kernels["channel_mipmap"] = clCreateKernel(m_program, "channel_mipmap", &err);
	CHECK_ERROR_OCL(err, "creating channel_mipmap kernel", return false);

	//this kernel computes log average luminance of the image
	kernels["finalReduc"] = clCreateKernel(m_program, "finalReduc", &err);
	CHECK_ERROR_OCL(err, "creating finalReduc kernel", return false);

	//performs the reinhard global tone mapping operator
	kernels["reinhardLocal"] = clCreateKernel(m_program, "reinhardLocal", &err);
	CHECK_ERROR_OCL(err, "creating reinhardLocal kernel", return false);


	/////////////////////////////////////////////////////////////////kernel sizes

	kernel2DSizes("computeLogAvgLum");
	kernel2DSizes("channel_mipmap");
	kernel2DSizes("reinhardLocal");

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

	//initialising information regarding all mipmap levels
	m_width = (int*) calloc(8, sizeof(int));
	m_height = (int*) calloc(8, sizeof(int));
	m_offset = (int*) calloc(8, sizeof(int));

	m_offset[0] = 0;
	m_width[0]  = image_width;
	m_height[0] = image_height;

	for (int level=1; level<num_mipmaps; level++) {
		m_width[level]  = m_width[level-1]/2;
		m_height[level] = m_height[level-1]/2;
		m_offset[level] = m_offset[level-1] + m_width[level-1]*m_height[level-1];
	}

	mems["logLum_Mips"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*image_width*image_height*2, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logLum_Mip1 memory", return false);

	mems["m_width"] = clCreateBuffer(m_clContext, CL_MEM_COPY_HOST_PTR, sizeof(int)*num_mipmaps, m_width, &err);
	CHECK_ERROR_OCL(err, "creating m_width memory", return false);

	mems["m_height"] = clCreateBuffer(m_clContext, CL_MEM_COPY_HOST_PTR, sizeof(int)*num_mipmaps, m_height, &err);
	CHECK_ERROR_OCL(err, "creating m_height memory", return false);

	mems["m_offset"] = clCreateBuffer(m_clContext, CL_MEM_COPY_HOST_PTR, sizeof(int)*num_mipmaps, m_offset, &err);
	CHECK_ERROR_OCL(err, "creating m_offset memory", return false);

	mems["logAvgLum"] = clCreateBuffer(m_clContext, CL_MEM_READ_WRITE, sizeof(float)*num_wg, NULL, &err);
	CHECK_ERROR_OCL(err, "creating logAvgLum memory", return false);

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
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 1, sizeof(cl_mem), &mems["logLum_Mips"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 2, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["computeLogAvgLum"], 3, sizeof(float*)*local_sizes["computeLogAvgLum"][0]*local_sizes["computeLogAvgLum"][1], NULL);
	CHECK_ERROR_OCL(err, "setting computeLogAvgLum arguments", return false);

	err  = clSetKernelArg(kernels["channel_mipmap"], 0, sizeof(cl_mem), &mems["logLum_Mips"]);
	CHECK_ERROR_OCL(err, "setting channel_mipmap arguments", return false);

	err  = clSetKernelArg(kernels["finalReduc"], 0, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["finalReduc"], 1, sizeof(unsigned int), &num_wg);
	CHECK_ERROR_OCL(err, "setting finalReduc arguments", return false);

	err  = clSetKernelArg(kernels["reinhardLocal"], 0, sizeof(cl_mem), &mem_images[0]);
	err  = clSetKernelArg(kernels["reinhardLocal"], 1, sizeof(cl_mem), &mem_images[1]);
	err  = clSetKernelArg(kernels["reinhardLocal"], 2, sizeof(cl_mem), &mems["logLum_Mips"]);
	err  = clSetKernelArg(kernels["reinhardLocal"], 3, sizeof(cl_mem), &mems["m_width"]);
	err  = clSetKernelArg(kernels["reinhardLocal"], 4, sizeof(cl_mem), &mems["m_height"]);
	err  = clSetKernelArg(kernels["reinhardLocal"], 5, sizeof(cl_mem), &mems["m_offset"]);
	err  = clSetKernelArg(kernels["reinhardLocal"], 6, sizeof(cl_mem), &mems["logAvgLum"]);
	err  = clSetKernelArg(kernels["reinhardLocal"], 7, sizeof(float), &key);
	err  = clSetKernelArg(kernels["reinhardLocal"], 8, sizeof(float), &sat);
	CHECK_ERROR_OCL(err, "setting globalTMO arguments", return false);


	reportStatus("\n\n");

	return true;
}

double ReinhardLocal::runCLKernels() {
	double start = omp_get_wtime();

	cl_int err;
	err = clEnqueueNDRangeKernel(m_queue, kernels["computeLogAvgLum"], 2, NULL, global_sizes["computeLogAvgLum"], local_sizes["computeLogAvgLum"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing computeLogAvgLum kernel", return false);

	err = clEnqueueNDRangeKernel(m_queue, kernels["finalReduc"], 1, NULL, &global_sizes["finalReduc"][0], &local_sizes["finalReduc"][0], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing finalReduc kernel", return false);

	//creating mipmaps
	for (int level=1; level<num_mipmaps; level++) {
		err  = clSetKernelArg(kernels["channel_mipmap"], 1, sizeof(int), &m_width[level-1]);
		err  = clSetKernelArg(kernels["channel_mipmap"], 2, sizeof(int), &m_offset[level-1]);
		err  = clSetKernelArg(kernels["channel_mipmap"], 3, sizeof(int), &m_width[level]);
		err  = clSetKernelArg(kernels["channel_mipmap"], 4, sizeof(int), &m_height[level]);
		err  = clSetKernelArg(kernels["channel_mipmap"], 5, sizeof(int), &m_offset[level]);
		CHECK_ERROR_OCL(err, "setting channel_mipmap arguments", return false);

		err = clEnqueueNDRangeKernel(m_queue, kernels["channel_mipmap"], 2, NULL, global_sizes["channel_mipmap"], local_sizes["channel_mipmap"], 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "enqueuing channel_mipmap kernel", return false);
	}

	err = clEnqueueNDRangeKernel(m_queue, kernels["reinhardLocal"], 2, NULL, global_sizes["reinhardLocal"], local_sizes["reinhardLocal"], 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing transfer_data kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);
	return omp_get_wtime() - start;	
}


bool ReinhardLocal::runOpenCL(int input_texid, int output_texid) {
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
bool ReinhardLocal::runOpenCL(Image input, Image output) {

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


bool ReinhardLocal::cleanupOpenCL() {
	clReleaseMemObject(mem_images[0]);
	clReleaseMemObject(mem_images[1]);
	clReleaseMemObject(mems["logLum_Mips"]);
	clReleaseMemObject(mems["m_width"]);
	clReleaseMemObject(mems["m_height"]);
	clReleaseMemObject(mems["m_offset"]);
	clReleaseMemObject(mems["logAvgLum"]);
	clReleaseKernel(kernels["computeLogAvgLum"]);
	clReleaseKernel(kernels["channel_mipmap"]);
	clReleaseKernel(kernels["finalReduc"]);
	clReleaseKernel(kernels["reinhardLocal"]);
	releaseCL();
	return true;
}


bool ReinhardLocal::runReference(Image input, Image output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*NUM_CHANNELS);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");


	float logAvgLum = 0.f;

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float3 rgb = {getPixel(input, x, y, 0),
				getPixel(input, x, y, 1), getPixel(input, x, y, 2)};

			logAvgLum += log(getPixelLuminance(rgb) + 0.000001);
		}
	}
	logAvgLum = exp(logAvgLum/(input.width*input.height));

	float factor = key/logAvgLum;
	float scale[num_mipmaps-1];

	Image* mipmap_pyramid = (Image*) calloc(num_mipmaps, sizeof(Image));
	mipmap_pyramid[0] = input;
	for (int i=1; i<num_mipmaps; i++) {
		mipmap_pyramid[i] = image_mipmap(input, i);
		scale[i-1] = pow(2, i-1);
	}

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float local_logAvgLum = 0.f;
			int centre_x = 0;
			int centre_y = 0;
			int surround_x = x;
			int surround_y = y;

			for (int i=0; i<num_mipmaps-1; i++) {
				centre_x = surround_x;
				centre_y = surround_y;
				surround_x = centre_x/2;
				surround_y = centre_y/2;

				float3 centre_pixel = { getPixel(mipmap_pyramid[i], centre_x, centre_y, 0)/((float)PIXEL_RANGE),
										getPixel(mipmap_pyramid[i], centre_x, centre_y, 1)/((float)PIXEL_RANGE),
										getPixel(mipmap_pyramid[i], centre_x, centre_y, 2)/((float)PIXEL_RANGE)};
				float3 surround_pixel= {getPixel(mipmap_pyramid[i+1], surround_x, surround_y, 0)/((float)PIXEL_RANGE),
										getPixel(mipmap_pyramid[i+1], surround_x, surround_y, 1)/((float)PIXEL_RANGE),
										getPixel(mipmap_pyramid[i+1], surround_x, surround_y, 2)/((float)PIXEL_RANGE)};
				float centre_logAvgLum = getPixelLuminance(centre_pixel)*(key/logAvgLum);
				float surround_logAvgLum = getPixelLuminance(surround_pixel)*(key/logAvgLum);


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