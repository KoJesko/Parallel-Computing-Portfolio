/* image_io.h - shared helpers for the native (real MPI / CUDA) versions.
 *
 * Image text format (same file the C* simulator program reads):
 *     rows cols
 *     k00 k01 k02 k10 k11 k12 k20 k21 k22      3x3 mask
 *     divisor offset
 *     rows*cols pixels (0..255)
 * The reader also accepts plain PGM (P2) files such as the simulator's
 * output, in which case the mask fields are left at "identity".
 */
#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
    int rows, cols;
    int kern[9];
    int divisor, offset;
    unsigned char *pix;     /* rows*cols */
} Image;

typedef struct { const char *name; int kern[9]; int divisor, offset; } FilterDef;

static const FilterDef FILTERS[] = {
    { "smooth",  { 1, 2, 1,  2, 4, 2,  1, 2, 1 }, 16,   0 },  /* Gaussian blur   */
    { "box",     { 1, 1, 1,  1, 1, 1,  1, 1, 1 },  9,   0 },  /* mean filter     */
    { "sharpen", { 0,-1, 0, -1, 5,-1,  0,-1, 0 },  1,   0 },
    { "edge",    {-1,-1,-1, -1, 8,-1, -1,-1,-1 },  1,   0 },  /* Laplacian       */
    { "emboss",  {-2,-1, 0, -1, 1, 1,  0, 1, 2 },  1, 128 },
};
#define NFILTERS ((int)(sizeof(FILTERS) / sizeof(FILTERS[0])))

static const FilterDef *find_filter(const char *name) {
    for (int i = 0; i < NFILTERS; i++)
        if (strcmp(FILTERS[i].name, name) == 0) return &FILTERS[i];
    return NULL;
}

static void set_filter(Image *im, const FilterDef *f) {
    memcpy(im->kern, f->kern, sizeof im->kern);
    im->divisor = f->divisor;
    im->offset = f->offset;
}

/* skip whitespace and '#' comments */
static int read_int(FILE *fp, int *v) {
    int ch;
    for (;;) {
        ch = fgetc(fp);
        if (ch == EOF) return 0;
        if (ch == '#') { while (ch != '\n' && ch != EOF) ch = fgetc(fp); continue; }
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') continue;
        ungetc(ch, fp);
        return fscanf(fp, "%d", v) == 1;
    }
}

static int read_header_and_pixels(FILE *fp, Image *im) {
    int c0 = fgetc(fp), c1 = fgetc(fp);
    int is_pgm = (c0 == 'P' && c1 == '2');
    if (!is_pgm) { ungetc(c1, fp); ungetc(c0, fp); }
    if (is_pgm) {
        int maxv;
        static const int id[9] = { 0,0,0, 0,1,0, 0,0,0 };
        if (!read_int(fp, &im->cols) || !read_int(fp, &im->rows) || !read_int(fp, &maxv)) return 0;
        memcpy(im->kern, id, sizeof id); im->divisor = 1; im->offset = 0;
    } else {
        if (!read_int(fp, &im->rows) || !read_int(fp, &im->cols)) return 0;
        for (int i = 0; i < 9; i++) if (!read_int(fp, &im->kern[i])) return 0;
        if (!read_int(fp, &im->divisor) || !read_int(fp, &im->offset)) return 0;
        if (im->divisor == 0) im->divisor = 1;
    }
    size_t n = (size_t)im->rows * im->cols;
    im->pix = (unsigned char *)malloc(n);
    for (size_t i = 0; i < n; i++) {
        int v;
        if (!read_int(fp, &v)) return 0;
        im->pix[i] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
    }
    return 1;
}

static int read_image(const char *path, Image *im) {
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 0; }
    int ok = read_header_and_pixels(fp, im);
    fclose(fp);
    if (!ok) fprintf(stderr, "bad image file %s\n", path);
    return ok;
}

/* deterministic synthetic test scene: gradient, shapes, stripes, noise */
static void gen_image(Image *im, int rows, int cols, unsigned seed) {
    im->rows = rows; im->cols = cols;
    im->pix = (unsigned char *)malloc((size_t)rows * cols);
    unsigned s = seed ? seed : 12345u;
    for (int y = 0; y < rows; y++)
        for (int x = 0; x < cols; x++) {
            double fx = (double)x / cols, fy = (double)y / rows;
            double v = 40 + 120 * fx * (1 - 0.5 * fy);                         /* gradient  */
            double dx = fx - 0.32, dy = fy - 0.38;
            if (dx * dx + dy * dy < 0.045) v = 220;                            /* disc      */
            if (fx > 0.58 && fx < 0.88 && fy > 0.18 && fy < 0.48) v = 25;      /* square    */
            if (fy > 0.62 && fy < 0.9 && fabs(fx - 0.7) < (fy - 0.62) * 0.7) v = 180; /* triangle */
            if (fy > 0.66 && fy < 0.92 && fx > 0.08 && fx < 0.45)              /* stripes   */
                v = ((int)(fx * 64) % 2) ? 235 : 60;
            s = s * 1103515245u + 12345u;                                      /* noise     */
            v += (double)((s >> 16) % 41) - 20;
            im->pix[(size_t)y * cols + x] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
}

static int write_text(const char *path, const Image *im) {
    FILE *fp = fopen(path, "w");
    if (!fp) return 0;
    fprintf(fp, "%d %d\n", im->rows, im->cols);
    for (int i = 0; i < 9; i++) fprintf(fp, "%d%c", im->kern[i], (i % 3 == 2) ? '\n' : ' ');
    fprintf(fp, "%d %d\n", im->divisor, im->offset);
    for (int y = 0; y < im->rows; y++) {
        for (int x = 0; x < im->cols; x++)
            fprintf(fp, "%d%c", im->pix[(size_t)y * im->cols + x], x + 1 < im->cols ? ' ' : '\n');
    }
    fclose(fp);
    return 1;
}

/* 8-bit grayscale BMP (opens in any Windows image viewer) */
static int write_bmp(const char *path, const unsigned char *pix, int rows, int cols) {
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;
    int stride = (cols + 3) & ~3;
    unsigned dataSize = (unsigned)stride * rows, off = 14 + 40 + 1024;
    unsigned fileSize = off + dataSize;
    unsigned char h[54] = { 'B','M' };
    memcpy(h + 2, &fileSize, 4); memcpy(h + 10, &off, 4);
    unsigned v40 = 40; memcpy(h + 14, &v40, 4);
    memcpy(h + 18, &cols, 4); memcpy(h + 22, &rows, 4);
    h[26] = 1; h[28] = 8;
    memcpy(h + 34, &dataSize, 4);
    unsigned n256 = 256; memcpy(h + 46, &n256, 4);
    fwrite(h, 1, 54, fp);
    for (int i = 0; i < 256; i++) { unsigned char q[4] = { (unsigned char)i, (unsigned char)i, (unsigned char)i, 0 }; fwrite(q, 1, 4, fp); }
    unsigned char *line = (unsigned char *)calloc(stride, 1);
    for (int y = rows - 1; y >= 0; y--) {
        memcpy(line, pix + (size_t)y * cols, cols);
        fwrite(line, 1, stride, fp);
    }
    free(line);
    fclose(fp);
    return 1;
}

/* reference single-threaded filter, identical arithmetic to every version */
static void filter_reference(const Image *im, unsigned char *out) {
    int R = im->rows, C = im->cols;
    for (int y = 0; y < R; y++)
        for (int x = 0; x < C; x++) {
            int sum = 0;
            for (int i = -1; i <= 1; i++) {
                int yy = y + i; yy = yy < 0 ? 0 : yy >= R ? R - 1 : yy;
                for (int j = -1; j <= 1; j++) {
                    int xx = x + j; xx = xx < 0 ? 0 : xx >= C ? C - 1 : xx;
                    sum += im->kern[(i + 1) * 3 + j + 1] * im->pix[(size_t)yy * C + xx];
                }
            }
            int v = sum / im->divisor + im->offset;   /* C truncation, same as C* */
            out[(size_t)y * C + x] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
}

static unsigned long long checksum(const unsigned char *p, size_t n) {
    unsigned long long h = 1469598103934665603ULL;     /* FNV-1a */
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

#endif
