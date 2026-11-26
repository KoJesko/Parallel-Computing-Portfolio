/* imagetool - generate sample input files, convert results to BMP, verify.
 *
 *   imagetool gen <rows> <cols> <filter> <out.txt> [seed]
 *        filters: smooth box sharpen edge emboss
 *   imagetool tobmp <image.txt | image.pgm> <out.bmp>
 *   imagetool verify <input.txt> <result.pgm>
 *        re-filters the input sequentially and compares every pixel
 */
#include "image_io.h"

int main(int argc, char **argv) {
    if (argc >= 6 && strcmp(argv[1], "gen") == 0) {
        Image im;
        const FilterDef *f = find_filter(argv[4]);
        if (!f) { fprintf(stderr, "unknown filter %s\n", argv[4]); return 1; }
        gen_image(&im, atoi(argv[2]), atoi(argv[3]), argc > 6 ? (unsigned)atoi(argv[6]) : 0);
        set_filter(&im, f);
        if (!write_text(argv[5], &im)) return 1;
        printf("wrote %s (%dx%d, filter %s)\n", argv[5], im.rows, im.cols, f->name);
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "tobmp") == 0) {
        Image im;
        if (!read_image(argv[2], &im)) return 1;
        if (!write_bmp(argv[3], im.pix, im.rows, im.cols)) return 1;
        printf("wrote %s (%dx%d)\n", argv[3], im.rows, im.cols);
        return 0;
    }
    if (argc == 4 && strcmp(argv[1], "verify") == 0) {
        Image in, res;
        if (!read_image(argv[2], &in) || !read_image(argv[3], &res)) return 1;
        if (in.rows != res.rows || in.cols != res.cols) { printf("FAIL: size mismatch\n"); return 1; }
        size_t n = (size_t)in.rows * in.cols, bad = 0;
        unsigned char *ref = (unsigned char *)malloc(n);
        filter_reference(&in, ref);
        for (size_t i = 0; i < n; i++) bad += ref[i] != res.pix[i];
        printf("%s: %zu of %zu pixels differ from sequential reference -> %s\n",
               argv[3], bad, n, bad ? "FAIL" : "PASS");
        return bad != 0;
    }
    fprintf(stderr, "usage:\n  imagetool gen <rows> <cols> <filter> <out.txt> [seed]\n"
                    "  imagetool tobmp <image.txt|.pgm> <out.bmp>\n"
                    "  imagetool verify <input.txt> <result.pgm>\n");
    return 1;
}
