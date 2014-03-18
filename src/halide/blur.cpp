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
  Func blur_x("blur_x"), blur_y("blur_y");
  Var c("c"), x("x"), y("y");

  // Algorithm
  clamped(x, y, c) = input(
    clamp(x, 0, input.width()-1),
    clamp(y, 0, input.height()-1),
    c) / 255.f;
  blur_x(x, y, c) = (
    clamped(x-2, y, c) +
    clamped(x-1, y, c) +
    clamped(x,   y, c) +
    clamped(x+1, y, c) +
    clamped(x+2, y, c)
    )/ 5.f;
  blur_y(x, y, c) = u8((
    blur_x(x, y-2, c) +
    blur_x(x, y-1, c) +
    blur_x(x, y,   c) +
    blur_x(x, y+1, c) +
    blur_x(x, y+2, c)
    ) / 5.f * 255);

  // Channel order
  input.set_stride(0, 4);
  input.set_extent(2, 4);
  blur_y.reorder_storage(c, x, y);
  blur_y.output_buffer().set_stride(0, 4);
  blur_y.output_buffer().set_extent(2, 4);

  // Schedules
  if (!strcmp(argv[1], "cpu"))
  {
    blur_y.parallel(y).vectorize(c, 4);
  }
  else if (!strcmp(argv[1], "gpu"))
  {
    blur_y.cuda_tile(x, y, 16, 4);
  }
  else
  {
    cout << "Invalid schedule type '" << argv[1] << "'" << endl;
    return 1;
  }

  compile(blur_y, input, argv[2], argv[3]);

  return 0;
}
