#include <string.h>
#include <cstdio>
#include <algorithm>
#include <omp.h>

#include "Stitching.h"
#include "opencl/stitching.h"
#if ENABLE_HALIDE
#include "halide/blur_cpu.h"
#include "halide/blur_gpu.h"
#endif

/* 
Combines different exposure images to create HDR, as proposed here:
http://www.ceng.metu.edu.tr/~akyuz/files/hdrgpu.pdf
*/

using namespace hdr;

Stitching::Stitching() : Filter() {
	m_name = "Stitching";
	m_type = STITCH;
}

bool Stitching::runHalideCPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool Stitching::runHalideGPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool Stitching::runOpenCL(LDRI input, Image output, const Params& params) {
	const size_t local = 1000;
	const size_t global = ceil((float)input.width*input.height/(float)local) * local;

	char flags[1024];
	sprintf(flags, "-cl-fast-relaxed-math -Dnum_images=%d -Dimage_size=%lu\n", input.numImages, input.width*input.height);

	if (!initCL(params, stitching_kernel, flags)) {
		return false;
	}

	cl_int err;
	cl_kernel k_stitch;
	cl_mem mem_LDRimages, mem_exposures, mem_HDRimage;

	//set up kernels
	k_stitch = clCreateKernel(m_program, "stitch", &err);
	CHECK_ERROR_OCL(err, "creating stitch kernel", return false);

	mem_LDRimages = clCreateBuffer(m_context, CL_MEM_READ_WRITE,
		sizeof(float)*input.width*input.height*input.numImages*4, NULL, &err);
	CHECK_ERROR_OCL(err, "creating LDRimages memory", return false);

	mem_exposures = clCreateBuffer(m_context, CL_MEM_READ_WRITE, sizeof(float)*input.numImages, NULL, &err);
	CHECK_ERROR_OCL(err, "creating exposures memory", return false);

	mem_HDRimage = clCreateBuffer(m_context, CL_MEM_READ_WRITE, 
		sizeof(float)*input.width*input.height*4, NULL, &err);
	CHECK_ERROR_OCL(err, "creating HDRimage memory", return false);


	for (int i=0; i<input.numImages; i++) {
		err = clEnqueueWriteBuffer(m_queue, mem_LDRimages, CL_TRUE,
			i*sizeof(float)*input.width*input.height*4, 
			sizeof(float)*input.width*input.height*4, input.images[i].data, 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "writing LDRimage memory", return false);

		err = clEnqueueWriteBuffer(m_queue, mem_exposures, CL_TRUE,
			i*sizeof(float), sizeof(float), &input.images[i].exposure, 0, NULL, NULL);
		CHECK_ERROR_OCL(err, "writing exposure memory", return false);
	}

	err  = clSetKernelArg(k_stitch, 0, sizeof(cl_mem), &mem_LDRimages);
	err  = clSetKernelArg(k_stitch, 1, sizeof(cl_mem), &mem_exposures);
	err  = clSetKernelArg(k_stitch, 2, sizeof(cl_mem), &mem_HDRimage);
	CHECK_ERROR_OCL(err, "setting stitch arguments", return false);


	//let it begin
	double start = omp_get_wtime();

	err = clEnqueueNDRangeKernel(m_queue, k_stitch, 1, NULL, &global, &local, 0, NULL, NULL);
	CHECK_ERROR_OCL(err, "enqueuing stitch kernel", return false);

	err = clFinish(m_queue);
	CHECK_ERROR_OCL(err, "running kernels", return false);
	double runTime = omp_get_wtime() - start;


	//read results back
	err = clEnqueueReadBuffer(m_queue, mem_HDRimage,
		CL_TRUE, 0, sizeof(float)*output.width*output.height*4, output.data, 0, NULL, NULL );
	CHECK_ERROR_OCL(err, "reading HDRimage memory", return false);

	reportStatus("Finished OpenCL kernel");


	bool passed = verify(input, output);
	reportStatus(
		"Finished in %lf ms (verification %s)",
		runTime*1000, passed ? "passed" : "failed");

	//cleanup
	clReleaseMemObject(mem_LDRimages);
	clReleaseMemObject(mem_exposures);
	clReleaseMemObject(mem_HDRimage);
	clReleaseKernel(k_stitch);
	releaseCL();
	return passed;
}


bool Stitching::runReference(LDRI input, Image output) {
	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	float3 hdr, ldr;

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float weightedSum = 0;
			hdr.x = hdr.y = hdr.z = 0;
			for (int i=0; i < input.numImages; i++) {
				ldr.x = getPixel(input.images[i], x, y, 0)*255.f;
				ldr.y = getPixel(input.images[i], x, y, 1)*255.f;
				ldr.z = getPixel(input.images[i], x, y, 2)*255.f;

				float luminance = getPixelLuminance(ldr);
				float w = weight(luminance);
				float exposure = input.images[i].exposure;

				hdr.x += (ldr.x/exposure) * w;
				hdr.y += (ldr.y/exposure) * w;
				hdr.z += (ldr.z/exposure) * w;

				weightedSum += w;
			}

			hdr.x = hdr.x/(weightedSum + 0.000001);
			hdr.y = hdr.y/(weightedSum + 0.000001);
			hdr.z = hdr.z/(weightedSum + 0.000001);

			output.data[(x + y*input.width)*4 + 0] = hdr.x/255.f;
			output.data[(x + y*input.width)*4 + 1] = hdr.y/255.f;
			output.data[(x + y*input.width)*4 + 2] = hdr.z/255.f;
			output.data[(x + y*input.width)*4 + 3] = getPixelLuminance(hdr);
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
