#include "Filter.h"

namespace hdr
{
class ReinhardGlobal : public Filter {
public:
	ReinhardGlobal();

	virtual bool runHalideCPU(Image input, Image output, const Params& params);
	virtual bool runHalideGPU(Image input, Image output, const Params& params);
	virtual bool setupOpenCL(const Params& params, const int image_size);
	virtual bool runOpenCL(Image input, Image output, const Params& params);
	virtual bool cleanupOpenCL();
	virtual bool runReference(Image input, Image output);
};
}