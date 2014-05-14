#include "Filter.h"

namespace hdr
{
class GradDom : public Filter {
public:
	GradDom(float _adjust_alpha=0.1f, float _beta=0.85f, float _sat=0.5f);

	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params);
	virtual double runCLKernels(bool recomputeMapping);
	virtual bool runOpenCL(int input_texid, int output_texid, bool recomputeMapping);
	virtual bool runOpenCL(Image input, Image output, bool recomputeMapping);
	virtual bool cleanupOpenCL();
	virtual bool runReference(Image input, Image output);

	float* attenuate_func(float* lum, int width, int height);
	float* poissonSolver(float* lum, float* div_grad, int width, int height, float terminationCriterea=0.001);
	float* apply_constant(float* input_lum, float* output_lum, int width, int height);

protected:
	float adjust_alpha;
	float beta;
	float sat;

	//information regarding all mipmap levels
	int num_mipmaps;
	int* m_width;		//at index i this contains the width of the mipmap at index i
	int* m_height;		//at index i this contains the height of the mipmap at index i
	int* m_offset;		//at index i this contains the start point to store the mipmap at level i
	float* m_divider;		//at index i this contains the value the pixels of gradient magnitude are going to be divided by	

};
}