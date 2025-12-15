#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

void sobel_u8_omp(const uint8_t * restrict src,
                  uint8_t * restrict dst,
                  int width,
                  int height);

uint8_t* load_image(const char *filename, int *w, int *h) {
    int channels;
    uint8_t *img = stbi_load(filename, w, h, &channels, 1);
    if (!img) {
        printf("failed to load image: %s\n", filename);
        return NULL;
    }
    return img;
}

void save_image(const char *filename, const uint8_t *data, int w, int h) {
    stbi_write_jpg(filename, w, h, 1, data, 90);
}

int main(void) {
    int w, h;

    uint8_t *src = load_image("image2.jpg", &w, &h);
    uint8_t *dst = (uint8_t *) malloc((size_t)w * h);
    if (!src || !dst) return 1;

    for (int i = 0; i < 3; ++i) {
        sobel_u8_omp(src, dst, w, h);
    }

    const int iterations = 100;
    double t0 = omp_get_wtime();
    for (int i = 0; i < iterations; ++i) {
        sobel_u8_omp(src, dst, w, h);
    }
    double t1 = omp_get_wtime();

    printf("avg time (%d iterations): %.3f ms\n", iterations, (t1 - t0) * 1000.0 / iterations);
    printf("threads : %d\n", omp_get_max_threads());

    save_image("output.jpg", dst, w, h);

    free(src);
    free(dst);
    return 0;
}
