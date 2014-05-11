
float GL_to_CL(uint val);
float3 RGBtoXYZ(float3 rgb);

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

//use this one for android because android's opencl specification is buggy
/*kernel void transfer_data(__read_only image2d_t input_image, __global float* image) {
	int2 pos;
	uint4 pixel;
	for (pos.y = get_global_id(1); pos.y < height; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < width; pos.x += get_global_size(0)) {
			pixel = read_imageui(input_image, sampler, pos);
			image[(pos.x + pos.y*width)*NUM_CHANNELS + 0] = GL_to_CL(pixel.x);
			image[(pos.x + pos.y*width)*NUM_CHANNELS + 1] = GL_to_CL(pixel.y);
			image[(pos.x + pos.y*width)*NUM_CHANNELS + 2] = GL_to_CL(pixel.z);		
			image[(pos.x + pos.y*width)*NUM_CHANNELS + 3] = GL_to_CL(pixel.w);
		}
	}
}*/

kernel void transfer_data(__read_only image2d_t input_image, __global uchar* image) {
	int2 pos;
	uint4 pixel;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			pixel = read_imageui(input_image, sampler, pos);
			image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 0] = GL_to_CL(pixel.x);
			image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 1] = GL_to_CL(pixel.y);
			image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 2] = GL_to_CL(pixel.z);
			image[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 3] = GL_to_CL(pixel.w);
		}
	}
}


//this kernel computes logAvgLum and Lwhite by performing reduction
//the results are stored in an array of size num_work_groups
kernel void computeLogAvgLum( 	__global uchar* image,
								__global float* logAvgLum,
								__global float* Lwhite,
								__local float* Lwhite_loc,
								__local float* logAvgLum_loc) {


	const int gid = get_global_id(0);	//id in the entire global memory
	const int global_size = get_global_size(0);

	float lum;
	float Lwhite_acc = 0.f;		//maximum luminance in the image
	float logAvgLum_acc = 0.f;

	for (int i=gid; i < image_size; i+=global_size) {
		lum = image[i*NUM_CHANNELS + 0]*0.2126 + image[i*NUM_CHANNELS + 1]*0.7152 + image[i*NUM_CHANNELS + 2]*0.0722;

		Lwhite_acc = (lum > Lwhite_acc) ? lum : Lwhite_acc;
		logAvgLum_acc += log(lum + 0.000001);
	}


	const int lid = get_local_id(0);	//id within the work group
	Lwhite_loc[lid] = Lwhite_acc;
	logAvgLum_loc[lid] = logAvgLum_acc;

	// Perform parallel reduction
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int offset = get_local_size(0)/2; offset > 0; offset = offset/2) {
		if (lid < offset) {
			Lwhite_loc[lid] = (Lwhite_loc[lid+offset] > Lwhite_loc[lid]) ? Lwhite_loc[lid+offset] : Lwhite_loc[lid];
			logAvgLum_loc[lid] += logAvgLum_loc[lid + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	const int group_id = get_group_id(0);
	if (lid == 0) {
		Lwhite[group_id] = Lwhite_loc[0];
		logAvgLum[group_id] = logAvgLum_loc[0];
	}
}

kernel void global_TMO( __global uchar* image,
						__global float* logAvgLum_acc,
						__global float* Lwhite_acc,
						const float key,
						const float sat,
						const unsigned int num_reduc_bins) {


	float Lwhite = 0.f;
	float logAvgLum = 0.f;
	for (int i=0; i<num_reduc_bins; i++) {
		if (Lwhite < Lwhite_acc[i]) Lwhite = Lwhite_acc[i];
		logAvgLum += logAvgLum_acc[i];
	}
	logAvgLum = exp(logAvgLum/image_size);

	const int gid = get_global_id(0);
	if (gid < image_size) {
		float3 rgb, xyz;
		rgb.x = image[gid*NUM_CHANNELS + 0];
		rgb.y = image[gid*NUM_CHANNELS + 1];
		rgb.z = image[gid*NUM_CHANNELS + 2];

		xyz = RGBtoXYZ(rgb);

		float L  = (key/logAvgLum) * xyz.y;
		float Ld = (L * (1.0 + L/(Lwhite * Lwhite)) )/(1.0 + L);

		image[gid*NUM_CHANNELS + 0] = clamp(pow(rgb.x/xyz.y, sat) * Ld, 0.f, 1.f)*255.f;
		image[gid*NUM_CHANNELS + 1] = clamp(pow(rgb.y/xyz.y, sat) * Ld, 0.f, 1.f)*255.f;
		image[gid*NUM_CHANNELS + 2] = clamp(pow(rgb.z/xyz.y, sat) * Ld, 0.f, 1.f)*255.f;
	}
}

kernel void transfer_data_output(__write_only image2d_t output_image, __global uchar* output) {
	int2 pos;
	uint4 pixel;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			pixel.x = output[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 0];
			pixel.y = output[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 1];
			pixel.z = output[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 2];
			pixel.w = output[(pos.x + pos.y*WIDTH)*NUM_CHANNELS + 3];
			write_imageui(output_image, pos, pixel);
		}
	}
}


float3 RGBtoXYZ(float3 rgb) {
	float3 xyz;
	xyz.x = rgb.x*0.4124 + rgb.y*0.3576 + rgb.z*0.1805;
	xyz.y = rgb.x*0.2126 + rgb.y*0.7152 + rgb.z*0.0722;
	xyz.z = rgb.x*0.0193 + rgb.y*0.1192 + rgb.z*0.9505;
	return xyz;
}

float GL_to_CL(uint val) {
	if (val >= 14340) return round(0.1245790*val - 1658.44);	//>=128
	if (val >= 13316) return round(0.0622869*val - 765.408);	//>=64
	if (val >= 12292) return round(0.0311424*val - 350.800);	//>=32
	if (val >= 11268) return round(0.0155702*val - 159.443);	//>=16

	float v = (float) val;
	return round(0.0000000000000125922*pow(v,4.f) - 0.00000000026729*pow(v,3.f) + 0.00000198135*pow(v,2.f) - 0.00496681*v - 0.0000808829); 
}