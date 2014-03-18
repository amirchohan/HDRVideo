#include "common.h"
#include <iostream>

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    cout << "Usage: " << argv[0] << " cpu|gpu out_func out_prefix" << endl;
    return 1;
  }

  ImageParam input(UInt(8), 3, "input");
  Func clamped("clamped");
  Func convolved("convolved");
  Func sharpen("sharpen");
  Var c("c"), x("x"), y("y");

  // Algorithm
  clamped(x, y, c) = input(
    clamp(x, 0, input.width()-1),
    clamp(y, 0, input.height()-1),
    c) / 255.f;

  Image<int16_t> kernel(3, 3);
  kernel(0, 0) = -1;
  kernel(0, 1) = -1;
  kernel(0, 2) = -1;
  kernel(1, 0) = -1;
  kernel(1, 1) = 8;
  kernel(1, 2) = -1;
  kernel(2, 0) = -1;
  kernel(2, 1) = -1;
  kernel(2, 2) = -1;

  RDom r(kernel);
  convolved(x, y, c) += kernel(r.x, r.y) * clamped(x + r.x - 1, y + r.y - 1, c);
  sharpen(x, y, c) = u8(
    clamp(convolved(x, y, c)/8 + clamped(x, y, c), 0, 1) * 255
  );

  // Channel order
  input.set_stride(0, 4);
  input.set_extent(2, 4);
  sharpen.reorder_storage(c, x, y);
  sharpen.output_buffer().set_stride(0, 4);
  sharpen.output_buffer().set_extent(2, 4);

  // Schedules
  if (!strcmp(argv[1], "cpu"))
  {
    sharpen.parallel(y).vectorize(c, 4);
  }
  else if (!strcmp(argv[1], "gpu"))
  {
    sharpen.cuda_tile(x, y, 16, 4);
  }
  else
  {
    cout << "Invalid schedule type '" << argv[1] << "'" << endl;
    return 1;
  }

  compile(sharpen, input, argv[2], argv[3]);

  return 0;
}
