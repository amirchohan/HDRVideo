#include <string.h>
#include <cstdio>
#include <algorithm>
#include <omp.h>

#include "Reinhard.h"
//#include "opencl/toneMap.h"
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
	return false;
}


bool Reinhard::runReference(LDRI input, Image output) {

	// Check for cached result
	if (m_reference.data) {
		memcpy(output.data, m_reference.data, output.width*output.height*4);
		reportStatus("Finished reference (cached)");
		return true;
	}

	reportStatus("\tRunning reference");

	float* hdr_luminance = (float*) calloc(input.width * input.height, sizeof(float));

	float logAvgLum = 0;
	float3 hdr, ldr;
	float Ywhite = 0.f;

	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			float lum = getPixel(input.images[0], x, y, 3);
			if (lum > Ywhite) Ywhite = lum;

			logAvgLum += log(lum + 0.000001);
		}
	}

	float key = 1.0f;
	float sat = 1.5f;

	logAvgLum = exp(logAvgLum/(input.width*input.height));

	//global Tone-mapping operator
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



	reportStatus("\tFinished reference");

	// Cache result
	m_reference.width = output.width;
	m_reference.height = output.height;
	m_reference.data = new float[output.width*output.height*4];
	memcpy(m_reference.data, output.data, output.width*output.height*4);

	return true;
}
