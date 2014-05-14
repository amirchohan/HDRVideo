
float GL_to_CL(uint val);
float3 RGBtoXYZ(float3 rgb);

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

//this kernel computes logAvgLum and Lwhite by performing reduction
//the results are stored in an array of size num_work_groups
kernel void computeLogAvgLum( 	__read_only image2d_t image,
								__global float* logLum,
								__global float* logAvgLum,
								__local float* logAvgLum_loc) {

	float lum;
	float Lwhite_acc = 0.f;		//maximum luminance in the image
	float logAvgLum_acc = 0.f;

	int2 pos;
	uint4 pixel;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			pixel = read_imageui(image, sampler, pos);
			lum = GL_to_CL(pixel.x)*0.2126
				+ GL_to_CL(pixel.y)*0.7152
				+ GL_to_CL(pixel.z)*0.0722;

			Lwhite_acc = (lum > Lwhite_acc) ? lum : Lwhite_acc;
			logAvgLum_acc += log(lum + 0.000001);
			logLum[pos.x + pos.y*WIDTH] = (GL_to_CL(pixel.x)/255.f)*0.2126
										+ (GL_to_CL(pixel.y)/255.f)*0.7152
										+ (GL_to_CL(pixel.z)/255.f)*0.0722;
		}
	}

	pos.x = get_local_id(0);
	pos.y = get_local_id(1);
	const int lid = pos.x + pos.y*get_local_size(0);	//local id in one dimension
	logAvgLum_loc[lid] = logAvgLum_acc;

	// Perform parallel reduction
	barrier(CLK_LOCAL_MEM_FENCE);


	for(int offset = (get_local_size(0)*get_local_size(1))/2; offset > 0; offset = offset/2) {
		if (lid < offset) {
			logAvgLum_loc[lid] += logAvgLum_loc[lid + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	const int num_work_groups = get_global_size(0)/get_local_size(0);	//number of workgroups in x dim
	const int group_id = get_group_id(0) + get_group_id(1)*num_work_groups;
	if (lid == 0) {
		logAvgLum[group_id] = logAvgLum_loc[0];
	}
}

kernel void channel_mipmap(	__global float* mipmap,	//array containing all the mipmap levels
							const int prev_width,	//width of the previous mipmap
							const int prev_offset, 	//start point of the previous mipmap 
							const int m_width,		//width of the mipmap being generated
							const int m_height,		//height of the mipmap being generated
							const int m_offset) { 	//start point to store the current mipmap
	int2 pos;
	for (pos.y = get_global_id(1); pos.y < m_height; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < m_width; pos.x += get_global_size(0)) {
			int _x = 2*pos.x;
			int _y = 2*pos.y;
			mipmap[pos.x + pos.y*m_width + m_offset] = 	(mipmap[_x + _y*prev_width + prev_offset]
														+ mipmap[_x+1 + _y*prev_width + prev_offset]
														+ mipmap[_x + (_y+1)*prev_width + prev_offset]
														+ mipmap[(_x+1) + (_y+1)*prev_width + prev_offset])/4.f;
		}
	}
}

kernel void finalReduc(	__global float* logAvgLum_acc,
						const unsigned int num_reduc_bins) {
	if (get_global_id(0)==0) {

		float Lwhite = 0.f;
		float logAvgLum = 0.f;
	
		for (int i=0; i<num_reduc_bins; i++) {
			logAvgLum += logAvgLum_acc[i];
		}
		logAvgLum_acc[0] = exp(logAvgLum/((float)image_size));
	}
	else return;
}


kernel void reinhardLocal(	__global float* Ld_array,
							__global float* logLumMips,
							__global int* m_width,
							__global int* m_height,
							__global int* m_offset,
							__global float* logAvgLum_acc,
							const float key) {

	float factor = key/logAvgLum_acc[0];

	const float scale[7] = {1.f, 2.f, 4.f, 8.f, 16.f, 32.f, 64.f};
	int2 pos, centre_pos, surround_pos;
	uint4 pixel;
	float3 rgb, xyz;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			float local_logAvgLum = 0.f;
			surround_pos = pos;
			for (int i=0; i<NUM_MIPMAPS-1; i++) {
				centre_pos = surround_pos;
				surround_pos = centre_pos/2;

				float centre_logAvgLum = logLumMips[centre_pos.x + centre_pos.y*m_width[i] + m_offset[i]]*factor;
				float surround_logAvgLum = logLumMips[surround_pos.x + surround_pos.y*m_width[i+1] + m_offset[i+1]]*factor;

				float logAvgLum_diff = centre_logAvgLum - surround_logAvgLum;
				logAvgLum_diff = logAvgLum_diff >= 0 ? logAvgLum_diff : -logAvgLum_diff;

				if (logAvgLum_diff/(pow(2.f, (float)PHI)*key/(scale[i]*scale[i]) + centre_logAvgLum) > EPSILON) {
					local_logAvgLum = centre_logAvgLum;
					break;
				}
				else local_logAvgLum = surround_logAvgLum;
			}
			Ld_array[pos.x + pos.y*WIDTH] = factor /(1.0 + local_logAvgLum);
		}
	}
}

kernel void tonemap(__read_only image2d_t input_image,
					__write_only image2d_t output_image,
					__global float* Ld_array,
					const float sat) {

	int2 pos;
	uint4 pixel;
	float3 rgb, xyz;
	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {
		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {
			pixel = read_imageui(input_image, sampler, pos);
			rgb.x = GL_to_CL(pixel.x);
			rgb.y = GL_to_CL(pixel.y);
			rgb.z = GL_to_CL(pixel.z);

			xyz = RGBtoXYZ(rgb);

			float Ld  = Ld_array[pos.x + pos.y*WIDTH] * xyz.y;

			pixel.x = clamp((pow(rgb.x/xyz.y, sat)*Ld)*255.f, 0.f, 255.f);
			pixel.y = clamp((pow(rgb.y/xyz.y, sat)*Ld)*255.f, 0.f, 255.f);
			pixel.z = clamp((pow(rgb.z/xyz.y, sat)*Ld)*255.f, 0.f, 255.f);
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
	//return (float)val;
}