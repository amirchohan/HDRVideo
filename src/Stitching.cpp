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
}

bool Stitching::runHalideCPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool Stitching::runHalideGPU(LDRI input, Image output, const Params& params) {
	return false;
}

bool Stitching::runOpenCL(LDRI input, Image output, const Params& params) {
	return false;
}


bool Stitching::runReference(LDRI input, Image output) {
	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("Running reference");

	float* hdr_image = (float*) calloc(4*input.height*input.width, sizeof(float));

	float logAvgLum = 0;
	float3 hdr, ldr;
	float Ywhite = 0.f;

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float weightedSum = 0;
			float hdr_l;
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
			hdr_l = getPixelLuminance(hdr);

			hdr_image[(x + y*input.width)*4 + 0] = hdr.x;
			hdr_image[(x + y*input.width)*4 + 1] = hdr.y;
			hdr_image[(x + y*input.width)*4 + 2] = hdr.z;
			hdr_image[(x + y*input.width)*4 + 3] = log(hdr_l + 0.000001);

			if (hdr_image[(x + y*input.width)*4 + 3] > Ywhite)
				Ywhite = hdr_image[(x + y*input.width)*4 + 3];

			logAvgLum += hdr_image[(x + y*input.width)*4 + 3];
		}
	}

	float key = 1.0f;
	float sat = 1.5f;

	logAvgLum = exp(logAvgLum/(input.width*input.height));

	//global Tone-mapping operator
	/*for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {
			float3 rgb, xyz;
			rgb.x = hdr_image[(x + y*input.width)*4 + 0];
			rgb.y = hdr_image[(x + y*input.width)*4 + 1];
			rgb.z = hdr_image[(x + y*input.width)*4 + 2];

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
	}*/


	key = 0.35f;
	float factor = key/logAvgLum;
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
						v1 += hdr_image[(x + y*input.width)*4 + 3]*r1;


						r2 = (1/(phi*pow((level+1)*scale[level+1], 2)) ) *
							exp(- (pow(_x,2) + pow(_y,2))/pow((level+1)*scale[level+1], 2));
						v2 += hdr_image[(x + y*input.width)*4 + 3]*r2;

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
			rgb.x = hdr_image[(x + y*input.width)*4 + 0];
			rgb.y = hdr_image[(x + y*input.width)*4 + 1];
			rgb.z = hdr_image[(x + y*input.width)*4 + 2];

			xyz = RGBtoXYZ(rgb);

			float Y  = (key/logAvgLum) * xyz.y;
			float Yd = Y /(1.0 + La);

			rgb.x = pow(rgb.x/xyz.x, sat) * Yd;
			rgb.y = pow(rgb.y/xyz.y, sat) * Yd;
			rgb.z = pow(rgb.z/xyz.z, sat) * Yd;

			printf("%f, %f, %f\n", rgb.x, rgb.y, rgb.z);

			setPixel(output, x, y, 0, rgb.x);
			setPixel(output, x, y, 1, rgb.y);
			setPixel(output, x, y, 2, rgb.z);
		}
		printf("%d\n", y);
	}

	reportStatus("Finished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*3];
	memcpy(m_reference.data, output.data, output.width*output.height*3);

	return true;
}
