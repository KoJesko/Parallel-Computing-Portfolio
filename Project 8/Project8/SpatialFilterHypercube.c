/* PROGRAM SpatialFilterHypercube  -  Project 8                          */
/*                                                                       */
/* MPI-based parallel spatial filtering of a grayscale image on a        */
/* 7-dimensional hypercube (2^7 = 128 processors).                       */
/*                                                                       */
/* Run in the C* simulator:                                              */
/*     MPI ON                                                            */
/*     INPUT = image256_smooth.txt                                       */
/*     OUTPUT = out_smooth.pgm                                           */
/*     OPEN SpatialFilterHypercube.c                                     */
/*     RUN                                                               */
/*                                                                       */
/* Input file format (all integers, whitespace separated):               */
/*     rows cols                                                         */
/*     k00 k01 k02  k10 k11 k12  k20 k21 k22     (3x3 filter mask)       */
/*     divisor offset                                                    */
/*     rows*cols pixel values 0..255, row by row                         */
/* Output: the filtered image as a plain-text PGM (P2) file.             */
/*                                                                       */
/* Parallel algorithm                                                    */
/*  1. The image is split into 128 horizontal strips.  Strip s is given  */
/*     to hypercube node Gray(s) (reflected binary Gray code), so strips */
/*     s and s+1 always live on nodes that differ in exactly one bit,    */
/*     i.e. on directly connected hypercube neighbours.                  */
/*  2. Node 0 reads the header and broadcasts it with a 7-step binomial  */
/*     tree that only uses hypercube links (dimension 6 down to 0).      */
/*  3. Node 0 streams the pixels from the input file and sends each      */
/*     strip to its owner (no node ever stores the whole image, which    */
/*     keeps per-node memory at a few KB).                               */
/*  4. Every node exchanges one boundary ("halo") row with each of its   */
/*     two strip neighbours.  The exchange is done in two parallel       */
/*     phases (even/odd pairs), each a single hop on the hypercube.      */
/*  5. Every node applies the 3x3 mask to its own strip (edges are       */
/*     handled by replicating the border pixels).                        */
/*  6. Node 0 collects the filtered strips in order and writes the PGM.  */
/*                                                                       */
/* Note: the C* simulator does not allow bitwise operators, so the Gray  */
/* code XOR is computed with integer arithmetic.                         */

ARCHITECTURE HYPERCUBE(7);
#include <mpi.h>

#define DIM 7
#define P 128
#define MAXR 4
#define MAXC 512
#define LOCSIZE 3072
#define BUFSIZE 2048

int rank, nprocs, strip, rows, cols, myrows, divisor, offset, npix;
int up, down;
int hdr[13];
int kern[3][3];
int loc[LOCSIZE];
int res[BUFSIZE];
int buf[BUFSIZE];
int r, c, s, k, node;
float tStart, tBcast, tScatter, tHalo, tFilter, tGather;
float myFilter, sumFilter, maxFilter;
MPI_Status st;

/* bitwise XOR of two non-negative integers, without bitwise operators */
int XorInt(int a, int b) {
  int result, place;
  result = 0;
  place = 1;
  while (a > 0 || b > 0) {
    if (a % 2 != b % 2)
      result = result + place;
    a = a / 2;
    b = b / 2;
    place = place * 2;
  }
  return result;
}

/* reflected binary Gray code: strip number -> hypercube node */
int Gray(int n) {
  return XorInt(n, n / 2);
}

/* inverse Gray code: hypercube node -> strip number */
int InvGray(int g) {
  int n, t;
  n = g;
  t = g / 2;
  while (t > 0) {
    n = XorInt(n, t);
    t = t / 2;
  }
  return n;
}

/* broadcast hdr[] from node 0 along the 7 hypercube dimensions */
void HypercubeBcast() {
  int d, bit;
  bit = P / 2;
  for (d = DIM - 1; d >= 0; d--) {
    if (rank % (2 * bit) == 0)
      MPI_Send(&hdr[0], 13, MPI_INT, rank + bit, 0, MPI_COMM_WORLD);
    else if (rank % (2 * bit) == bit)
      MPI_Recv(&hdr[0], 13, MPI_INT, rank - bit, 0, MPI_COMM_WORLD, &st);
    bit = bit / 2;
  }
}

/* send my first own row to the strip above, my last own row to the   */
/* strip below; receive their rows into my halo rows 0 and myrows+1    */
void HaloExchange() {
  /* phase 1: pairs (even strip, strip+1) */
  if (strip % 2 == 0) {
    if (down >= 0) {
      MPI_Send(&loc[myrows * cols], cols, MPI_INT, down, 3, MPI_COMM_WORLD);
      MPI_Recv(&loc[(myrows + 1) * cols], cols, MPI_INT, down, 3,
               MPI_COMM_WORLD, &st);
    }
  }
  else {
    MPI_Recv(&loc[0], cols, MPI_INT, up, 3, MPI_COMM_WORLD, &st);
    MPI_Send(&loc[cols], cols, MPI_INT, up, 3, MPI_COMM_WORLD);
  }
  /* phase 2: pairs (odd strip, strip+1) */
  if (strip % 2 == 1) {
    if (down >= 0) {
      MPI_Send(&loc[myrows * cols], cols, MPI_INT, down, 4, MPI_COMM_WORLD);
      MPI_Recv(&loc[(myrows + 1) * cols], cols, MPI_INT, down, 4,
               MPI_COMM_WORLD, &st);
    }
  }
  else if (up >= 0) {
    MPI_Recv(&loc[0], cols, MPI_INT, up, 4, MPI_COMM_WORLD, &st);
    MPI_Send(&loc[cols], cols, MPI_INT, up, 4, MPI_COMM_WORLD);
  }
  /* image borders: replicate the edge row */
  if (up < 0)
    for (c = 0; c < cols; c++)
      loc[c] = loc[cols + c];
  if (down < 0)
    for (c = 0; c < cols; c++)
      loc[(myrows + 1) * cols + c] = loc[myrows * cols + c];
}

/* apply the 3x3 mask to my own rows 1..myrows of loc[] -> res[]      */
/* (border columns clamped once per column, mask fully unrolled)      */
void FilterStrip() {
  int a, b, d, cl, cr, sum, v, out;
  for (r = 1; r <= myrows; r++) {
    a = (r - 1) * cols;      /* row above   */
    b = r * cols;            /* centre row  */
    d = (r + 1) * cols;      /* row below   */
    out = (r - 1) * cols;
    for (c = 0; c < cols; c++) {
      cl = c - 1;
      if (cl < 0) cl = 0;
      cr = c + 1;
      if (cr >= cols) cr = cols - 1;
      sum = kern[0][0] * loc[a + cl] + kern[0][1] * loc[a + c] + kern[0][2] * loc[a + cr]
          + kern[1][0] * loc[b + cl] + kern[1][1] * loc[b + c] + kern[1][2] * loc[b + cr]
          + kern[2][0] * loc[d + cl] + kern[2][1] * loc[d + c] + kern[2][2] * loc[d + cr];
      v = sum / divisor + offset;
      if (v < 0) v = 0;
      if (v > 255) v = 255;
      res[out + c] = v;
    }
  }
}

void PrintRows(int count) {
  int idx;
  for (idx = 0; idx < count; idx++) {
    cout << buf[idx];
    if ((idx + 1) % 16 == 0 || (idx + 1) % cols == 0)
      cout << endl;  /* simulator limits output line length */
  }
}

main() {
  MPI_Init();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  tStart = MPI_Wtime();

  /* ---- 1. node 0 reads the header, hypercube broadcast ---- */
  if (rank == 0)
    for (k = 0; k < 13; k++)
      cin >> hdr[k];
  HypercubeBcast();
  rows = hdr[0];
  cols = hdr[1];
  for (r = 0; r < 3; r++)
    for (c = 0; c < 3; c++)
      kern[r][c] = hdr[2 + 3 * r + c];
  divisor = hdr[11];
  offset = hdr[12];
  if (divisor == 0) divisor = 1;
  myrows = rows / P;
  npix = myrows * cols;
  if (rank == 0 && (rows % P != 0 || myrows > MAXR || cols > MAXC))
    cout << "ERROR: rows must be a multiple of 128, rows <= 512, cols <= 512"
         << endl;

  /* strip owned by this node and its two strip neighbours */
  strip = InvGray(rank);
  if (strip > 0) up = Gray(strip - 1);
  else up = -1;
  if (strip < P - 1) down = Gray(strip + 1);
  else down = -1;
  tBcast = MPI_Wtime();

  /* ---- 2. distribute strips: strip s -> node Gray(s) ---- */
  if (rank == 0) {
    for (k = 0; k < npix; k++)
      cin >> loc[cols + k];
    for (s = 1; s < P; s++) {
      for (k = 0; k < npix; k++)
        cin >> buf[k];
      MPI_Send(&buf[0], npix, MPI_INT, Gray(s), 1, MPI_COMM_WORLD);
    }
  }
  else
    MPI_Recv(&loc[cols], npix, MPI_INT, 0, 1, MPI_COMM_WORLD, &st);
  tScatter = MPI_Wtime();

  /* ---- 3. halo exchange with hypercube neighbours ---- */
  HaloExchange();
  tHalo = MPI_Wtime();

  /* ---- 4. local 3x3 spatial filter ---- */
  FilterStrip();
  tFilter = MPI_Wtime();

  /* filter-phase work: sum and max over all 128 nodes */
  myFilter = tFilter - tHalo;
  MPI_Reduce(&myFilter, &sumFilter, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&myFilter, &maxFilter, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
  tFilter = MPI_Wtime();

  /* ---- 5. collect filtered strips in strip order, write PGM ---- */
  if (rank == 0) {
    cout << "P2" << endl;
    cout << cols << " " << rows << endl;
    cout << 255 << endl;
    for (k = 0; k < npix; k++)
      buf[k] = res[k];
    PrintRows(npix);
    for (s = 1; s < P; s++) {
      MPI_Recv(&buf[0], npix, MPI_INT, Gray(s), 2, MPI_COMM_WORLD, &st);
      PrintRows(npix);
    }
    tGather = MPI_Wtime();
    cout << "# ---- Program performance summary (simulated time units) ----"
         << endl;
    cout << "# image " << rows << " x " << cols << ", " << P
         << " processors (7-D hypercube), " << myrows
         << " rows per processor" << endl;
    cout << "# header broadcast  : " << tBcast - tStart << endl;
    cout << "# strip distribution: " << tScatter - tBcast << endl;
    cout << "# halo exchange     : " << tHalo - tScatter << endl;
    cout << "# filtering (+reduce): " << tFilter - tHalo << endl;
    cout << "# collect + output  : " << tGather - tFilter << endl;
    cout << "# total             : " << tGather - tStart << endl;
    cout << "# filter work, all nodes (sequential equivalent): " << sumFilter
         << endl;
    cout << "# filter work, slowest node                     : " << maxFilter
         << endl;
    cout << "# filter-phase speedup  : " << sumFilter / maxFilter << endl;
    cout << "# filter-phase efficiency: " << sumFilter / maxFilter / P
         << endl;
  }
  else
    MPI_Send(&res[0], npix, MPI_INT, 0, 2, MPI_COMM_WORLD);

  MPI_Finalize();
}
