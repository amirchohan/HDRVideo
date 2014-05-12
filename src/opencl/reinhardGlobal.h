const char *reinhardGlobal_kernel =
"\n"
"float GL_to_CL(uint val);\n"
"float3 RGBtoXYZ(float3 rgb);\n"
"\n"
"const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"
"\n"
"//this kernel computes logAvgLum and Lwhite by performing reduction\n"
"//the results are stored in an array of size num_work_groups\n"
"kernel void computeLogAvgLum( 	__read_only image2d_t image,\n"
"								__global float* logAvgLum,\n"
"								__global float* Lwhite,\n"
"								__local float* Lwhite_loc,\n"
"								__local float* logAvgLum_loc) {\n"
"\n"
"	float lum;\n"
"	float Lwhite_acc = 0.f;		//maximum luminance in the image\n"
"	float logAvgLum_acc = 0.f;\n"
"\n"
"	int2 pos;\n"
"	uint4 pixel;\n"
"	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {\n"
"		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {\n"
"			pixel = read_imageui(image, sampler, pos);\n"
"			lum = GL_to_CL(pixel.x)*0.2126\n"
"				+ GL_to_CL(pixel.y)*0.7152\n"
"				+ GL_to_CL(pixel.z)*0.0722;\n"
"\n"
"			Lwhite_acc = (lum > Lwhite_acc) ? lum : Lwhite_acc;\n"
"			logAvgLum_acc += log(lum + 0.000001);\n"
"		}\n"
"	}\n"
"\n"
"	pos.x = get_local_id(0);\n"
"	pos.y = get_local_id(1);\n"
"	const int lid = pos.x + pos.y*get_local_size(0);	//local id in one dimension\n"
"	Lwhite_loc[lid] = Lwhite_acc;\n"
"	logAvgLum_loc[lid] = logAvgLum_acc;\n"
"\n"
"	// Perform parallel reduction\n"
"	barrier(CLK_LOCAL_MEM_FENCE);\n"
"\n"
"\n"
"	for(int offset = (get_local_size(0)*get_local_size(1))/2; offset > 0; offset = offset/2) {\n"
"		if (lid < offset) {\n"
"			Lwhite_loc[lid] = (Lwhite_loc[lid+offset] > Lwhite_loc[lid]) ? Lwhite_loc[lid+offset] : Lwhite_loc[lid];\n"
"			logAvgLum_loc[lid] += logAvgLum_loc[lid + offset];\n"
"		}\n"
"		barrier(CLK_LOCAL_MEM_FENCE);\n"
"	}\n"
"\n"
"	const int num_work_groups = get_global_size(0)/get_local_size(0);	//number of workgroups in x dim\n"
"	const int group_id = get_group_id(0) + get_group_id(1)*num_work_groups;\n"
"	if (lid == 0) {\n"
"		Lwhite[group_id] = Lwhite_loc[0];\n"
"		logAvgLum[group_id] = logAvgLum_loc[0];\n"
"	}\n"
"}\n"
"\n"
"kernel void finalReduc(	__global float* logAvgLum_acc,\n"
"						__global float* Lwhite_acc,\n"
"						const unsigned int num_reduc_bins) {\n"
"	if (get_global_id(0)==0) {\n"
"\n"
"		float Lwhite = 0.f;\n"
"		float logAvgLum = 0.f;\n"
"	\n"
"		for (int i=0; i<num_reduc_bins; i++) {\n"
"			if (Lwhite < Lwhite_acc[i]) Lwhite = Lwhite_acc[i];\n"
"			logAvgLum += logAvgLum_acc[i];\n"
"		}\n"
"		Lwhite_acc[0] = Lwhite;\n"
"		logAvgLum_acc[0] = exp(logAvgLum/((float)image_size));\n"
"	}\n"
"	else return;\n"
"}\n"
"\n"
"kernel void reinhardGlobal(	__read_only image2d_t input_image,\n"
"							__write_only image2d_t output_image,\n"
"							__global float* logAvgLum_acc,\n"
"							__global float* Lwhite_acc,\n"
"							const float key,\n"
"							const float sat) {\n"
"	float Lwhite = Lwhite_acc[0];\n"
"	float logAvgLum = logAvgLum_acc[0];\n"
"\n"
"	int2 pos;\n"
"	uint4 pixel;\n"
"	float3 rgb, xyz;\n"
"	for (pos.y = get_global_id(1); pos.y < HEIGHT; pos.y += get_global_size(1)) {\n"
"		for (pos.x = get_global_id(0); pos.x < WIDTH; pos.x += get_global_size(0)) {\n"
"			pixel = read_imageui(input_image, sampler, pos);\n"
"\n"
"			rgb.x = GL_to_CL(pixel.x);\n"
"			rgb.y = GL_to_CL(pixel.y);\n"
"			rgb.z = GL_to_CL(pixel.z);\n"
"\n"
"			xyz = RGBtoXYZ(rgb);\n"
"\n"
"			float L\t= (key/logAvgLum) * xyz.y;\n"
"			float Ld = (L * (1.f + L/(Lwhite * Lwhite)) )/(1.f + L);\n"
"\n"
"			pixel.x = clamp((pow(rgb.x/xyz.y, sat)*Ld)*255.f, 0.f, 255.f);\n"
"			pixel.y = clamp((pow(rgb.y/xyz.y, sat)*Ld)*255.f, 0.f, 255.f);\n"
"			pixel.z = clamp((pow(rgb.z/xyz.y, sat)*Ld)*255.f, 0.f, 255.f);\n"
"			write_imageui(output_image, pos, pixel);\n"
"		}\n"
"	}\n"
"}\n"
"\n"
"\n"
"float3 RGBtoXYZ(float3 rgb) {\n"
"	float3 xyz;\n"
"	xyz.x = rgb.x*0.4124 + rgb.y*0.3576 + rgb.z*0.1805;\n"
"	xyz.y = rgb.x*0.2126 + rgb.y*0.7152 + rgb.z*0.0722;\n"
"	xyz.z = rgb.x*0.0193 + rgb.y*0.1192 + rgb.z*0.9505;\n"
"	return xyz;\n"
"}\n"
"\n"
"float GL_to_CL(uint val) {\n"
"	if (val >= 14340) return round(0.1245790*val - 1658.44);	//>=128\n"
"	if (val >= 13316) return round(0.0622869*val - 765.408);	//>=64\n"
"	if (val >= 12292) return round(0.0311424*val - 350.800);	//>=32\n"
"	if (val >= 11268) return round(0.0155702*val - 159.443);	//>=16\n"
"\n"
"	float v = (float) val;\n"
"	return round(0.0000000000000125922*pow(v,4.f) - 0.00000000026729*pow(v,3.f) + 0.00000198135*pow(v,2.f) - 0.00496681*v - 0.0000808829);\n"
"	//return (float)val;\n"
"}\n";
