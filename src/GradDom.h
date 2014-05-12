#include "Filter.h"

namespace hdr
{
class GradDom : public Filter {
public:
	GradDom();

	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params);
	virtual double runCLKernels();
	virtual bool runOpenCL(int input_texid, int output_texid);
	virtual bool runOpenCL(Image input, Image output);
	virtual bool cleanupOpenCL();
	virtual bool runReference(Image input, Image output);
};
}