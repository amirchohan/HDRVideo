#include "Filter.h"

namespace hdr
{
class HistEq : public Filter {
public:
	HistEq();

	virtual bool runHalideCPU(LDRI input, Image output, const Params& params);
	virtual bool runHalideGPU(LDRI input, Image output, const Params& params);
	virtual bool runOpenCL(LDRI input, Image output, const Params& params);
	virtual bool runReference(LDRI input, Image output);
};
}
