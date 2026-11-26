/* PROGRAM SpatialFilterSequential  -  Project 8 (sequential baseline)   */
/*                                                                       */
/* Same input / output / arithmetic as SpatialFilterHypercube.c, run on  */
/* a single processor, so the simulator's PARALLEL EXECUTION TIME of the */
/* two programs can be compared directly to get the real speedup.        */
/*                                                                       */
/*     INPUT = image256_smooth.txt                                       */
/*     OUTPUT = out_seq_smooth.pgm                                       */
/*     OPEN SpatialFilterSequential.c                                    */
/*     RUN                                                               */
/* (MPI mode is not needed for this program.)                           */
/*                                                                       */
/* Only a 3-row sliding window is kept in memory.                        */

#define MAXC 512

int win[3 * MAXC];
int outrow[MAXC];
int kern[3][3];
int rows, cols, divisor, offset;
int r, c, k, top, mid, bot, newest, cl, cr, sum, v, count;

void ReadRow(int slot) {
  for (c = 0; c < cols; c++)
    cin >> win[slot * cols + c];
}

void CopyRow(int src, int dst) {
  for (c = 0; c < cols; c++)
    win[dst * cols + c] = win[src * cols + c];
}

void FilterRow(int a, int b, int d) {
  a = a * cols;
  b = b * cols;
  d = d * cols;
  for (c = 0; c < cols; c++) {
    cl = c - 1;
    if (cl < 0) cl = 0;
    cr = c + 1;
    if (cr >= cols) cr = cols - 1;
    sum = kern[0][0] * win[a + cl] + kern[0][1] * win[a + c] + kern[0][2] * win[a + cr]
        + kern[1][0] * win[b + cl] + kern[1][1] * win[b + c] + kern[1][2] * win[b + cr]
        + kern[2][0] * win[d + cl] + kern[2][1] * win[d + c] + kern[2][2] * win[d + cr];
    v = sum / divisor + offset;
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    outrow[c] = v;
  }
  for (c = 0; c < cols; c++) {
    cout << outrow[c];
    if ((c + 1) % 16 == 0 || c + 1 == cols)
      cout << endl;
  }
}

main() {
  cin >> rows >> cols;
  for (r = 0; r < 3; r++)
    for (c = 0; c < 3; c++)
      cin >> kern[r][c];
  cin >> divisor >> offset;
  if (divisor == 0) divisor = 1;

  cout << "P2" << endl;
  cout << cols << " " << rows << endl;
  cout << 255 << endl;

  /* window slots rotate: top, mid, bot */
  top = 0;
  mid = 1;
  bot = 2;
  ReadRow(mid);
  CopyRow(mid, top);              /* replicate first row as border */
  for (r = 0; r < rows; r++) {
    if (r + 1 < rows)
      ReadRow(bot);
    else
      CopyRow(mid, bot);          /* replicate last row as border  */
    FilterRow(top, mid, bot);
    newest = top;                 /* rotate the window             */
    top = mid;
    mid = bot;
    bot = newest;
  }
}
