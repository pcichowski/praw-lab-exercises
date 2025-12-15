#include <stdint.h>
#include <string.h>
#include <omp.h>

static inline uint8_t clamp_u8_int(int v) {
    return (uint8_t)(v > 255 ? 255 : v);
}

void sobel_u8_omp(const uint8_t * restrict src,
                  uint8_t * restrict dst,
                  int width,
                  int height)
{
    if (!src || !dst || width <= 0 || height <= 0) return;

    // clear top bottom border
    memset(dst, 0, (size_t)width);
    memset(dst + (size_t)(height - 1) * width, 0, (size_t)width);

    if (width < 3 || height < 3) return;

    #pragma omp parallel for schedule(static)
    for (int y = 1; y < height - 1; ++y) {
        const uint8_t * restrict row0 = src + (size_t)(y - 1) * width;
        const uint8_t * restrict row1 = src + (size_t)(y) * width;
        const uint8_t * restrict row2 = src + (size_t)(y + 1) * width;
        
        uint8_t * restrict out = dst + (size_t)y * width;

        // clear left right border
        out[0] = 0;
        out[width - 1] = 0;

        #pragma omp simd
        for (int x = 1; x < width - 1; ++x) {
            // k00 k01 k02 
            // k10     k12
            // k20 k21 k22
            int k00 = row0[x - 1], k01 = row0[x], k02 = row0[x + 1];
            int k10 = row1[x - 1], k12 = row1[x + 1];
            int k20 = row2[x - 1], k21 = row2[x], k22 = row2[x + 1];

            int gx = (-k00 + k02) + (-2 * k10 + 2 * k12) + (-k20 + k22);
            int gy = (-k00 - 2 * k01 - k02) + (k20 + 2 * k21 + k22);

            int agx = gx < 0 ? -gx : gx;
            int agy = gy < 0 ? -gy : gy;

            int mag = agx + agy;
            out[x] = clamp_u8_int(mag);
        }
    }
}
