#include "Filter.h"

namespace hdr
{
class HistEq : public Filter {
public:
	HistEq();

	virtual bool runHalideCPU(Image input, Image output, const Params& params);
	virtual bool runHalideGPU(Image input, Image output, const Params& params);
	virtual bool runOpenCL(Image input, Image output, const Params& params);
	virtual bool setupOpenCL(const Params& params, const int image_size);
	virtual bool cleanupOpenCL();
	virtual bool runReference(Image input, Image output);
};
}
