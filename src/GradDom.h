#include "Filter.h"

namespace hdr
{
class GradDom : public Filter {
public:
	GradDom();

	virtual bool runHalideCPU(Image input, Image output, const Params& params);
	virtual bool runHalideGPU(Image input, Image output, const Params& params);
	virtual bool setupOpenCL(cl_context_properties context_prop[], const Params& params, const int image_size);
	virtual bool runOpenCL(Image input, Image output, const Params& params);
	virtual bool cleanupOpenCL();
	virtual bool runReference(Image input, Image output);
};
}