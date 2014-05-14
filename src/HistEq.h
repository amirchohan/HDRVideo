#include "Filter.h"

namespace hdr
{
class HistEq : public Filter {
public:
	HistEq();

	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params);
	virtual double runCLKernels(bool recomputeMapping);
	virtual bool runOpenCL(int input_texid, int output_texid, bool recomputeMapping);
	virtual bool runOpenCL(Image input, Image output, bool recomputeMapping);
	virtual bool cleanupOpenCL();
	virtual bool runReference(Image input, Image output);
};
}
