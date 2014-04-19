
float3 RGBtoXYZ(float3 rgb);

kernel void computeLogAvgLum( __global read_only float* image, __global float* logAvgLum, __global float* Ywhite) {
	//this kernel computes logAvgLum and Ywhite by performing reduction
	//the results are stored in an array of size num_work_groups

	int gid = get_global_id(0);	//id in the entire global memory

	float lum;
	float Ywhite_acc = 0.f;		//maximum luminance in the image
	float logAvgLum_acc = 0.f;
	while (gid < image_size) {
		lum = image[gid*4 + 0]*0.2126 + image[gid*4 + 1]*0.7152 + image[gid*4 + 2]*0.0722;

		Ywhite_acc = (lum > Ywhite_acc) ? lum : Ywhite_acc;
		logAvgLum_acc += log(lum + 0.000001);

		gid += get_global_size(0);
	}

	__local float Ywhite_loc[32];
	__local float logAvgLum_loc[32];

	int lid = get_local_id(0);	//id within the work group
	Ywhite_loc[lid] = Ywhite_acc;
	logAvgLum_loc[lid] = logAvgLum_acc;

	// Perform parallel reduction
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int offset = get_local_size(0)/2; offset > 0; offset = offset/2) {
		if (lid < offset) {
			Ywhite_loc[lid] = (Ywhite_loc[lid+offset] > Ywhite_loc[lid]) ? Ywhite_loc[lid+offset] : Ywhite_loc[lid];
			logAvgLum_loc[lid] += logAvgLum_loc[lid + offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (lid == 0) {
		Ywhite[get_group_id(0)] = Ywhite_loc[0];
		logAvgLum[get_group_id(0)] = logAvgLum_loc[0];
	}

}

kernel void global_TMO( __global float* input, __global float* output, __global float* logAvgLum_acc, __global float* Ywhite_acc) {
	float key = 1.0f;
	float sat = 1.5f;

	float logAvgLum = logAvgLum_acc[0]+logAvgLum_acc[1]+logAvgLum_acc[2]+logAvgLum_acc[3]+logAvgLum_acc[4]+logAvgLum_acc[5];
	logAvgLum = exp(logAvgLum/image_size);

	float Ywhite = 0.0f;
	for (int i=0; i<6; i++) {
		if (Ywhite < Ywhite_acc[i]) Ywhite = Ywhite_acc[i];
	}

	const int gid = get_global_id(0);
	if (gid < image_size) {
		float3 rgb, xyz;
		rgb.x = input[gid*4 + 0];
		rgb.y = input[gid*4 + 1];
		rgb.z = input[gid*4 + 2];

		xyz = RGBtoXYZ(rgb);

		float Y  = (key/logAvgLum) * xyz.y;
		float Yd = (Y * (1.0 + Y/(Ywhite * Ywhite)) )/(1.0 + Y);

		output[gid*4 + 0] = clamp(pow(rgb.x/xyz.y, sat) * Yd, 0.f, 1.f);
		output[gid*4 + 1] = clamp(pow(rgb.y/xyz.y, sat) * Yd, 0.f, 1.f);
		output[gid*4 + 2] = clamp(pow(rgb.z/xyz.y, sat) * Yd, 0.f, 1.f);
		output[gid*4 + 3] = 0.f;
	}
}

float3 RGBtoXYZ(float3 rgb) {
	float3 xyz;
	xyz.x = rgb.x*0.4124 + rgb.y*0.3576 + rgb.z*0.1805;
	xyz.y = rgb.x*0.2126 + rgb.y*0.7152 + rgb.z*0.0722;
	xyz.z = rgb.x*0.0193 + rgb.y*0.1192 + rgb.z*0.9505;
	return xyz;
}