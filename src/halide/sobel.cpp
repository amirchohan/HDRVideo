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
  Func clamped("clamped"), grayscale("grayscale");
  Func g_x("g_x"), g_y("g_y"), g_mag("g_mag");
  Func sobel("sobel");
  Var c("c"), x("x"), y("y");

  // Algorithm
  clamped(x, y, c) = input(
    clamp(x, 0, input.width()-1),
    clamp(y, 0, input.height()-1),
    c) / 255.f;
  grayscale(x, y) =
    clamped(x, y, 0)*0.299f +
    clamped(x, y, 1)*0.587f +
    clamped(x, y, 2)*0.114f;

  Image<int16_t> kernel(3, 3);
  kernel(0, 0) = -1;
  kernel(0, 1) = -2;
  kernel(0, 2) = -1;
  kernel(1, 0) = 0;
  kernel(1, 1) = 0;
  kernel(1, 2) = 0;
  kernel(2, 0) = 1;
  kernel(2, 1) = 2;
  kernel(2, 2) = 1;

  RDom r(kernel);
  g_x(x, y) += kernel(r.x, r.y) * grayscale(x + r.x - 1, y + r.y - 1);
  g_y(x, y) += kernel(r.y, r.x) * grayscale(x + r.x - 1, y + r.y - 1);
  g_mag(x, y) = sqrt(g_x(x, y)*g_x(x, y) + g_y(x, y)*g_y(x, y));
  sobel(x, y, c) = select(c==3, 255, u8(clamp(g_mag(x, y), 0, 1)*255));

  // Channel order
  input.set_stride(0, 4);
  input.set_extent(2, 4);
  sobel.reorder_storage(c, x, y);
  sobel.output_buffer().set_stride(0, 4);
  sobel.output_buffer().set_extent(2, 4);

  // Schedules
  if (!strcmp(argv[1], "cpu"))
  {
    sobel.parallel(y).vectorize(c, 4);
  }
  else if (!strcmp(argv[1], "gpu"))
  {
    sobel.cuda_tile(x, y, 16, 4);
  }
  else
  {
    cout << "Invalid schedule type '" << argv[1] << "'" << endl;
    return 1;
  }

  compile(sobel, input, argv[2], argv[3]);

  return 0;
}
