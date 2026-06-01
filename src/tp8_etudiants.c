// #define MODE_PHOTO // decommenter pour les images
#define MODE_PERF // decommenter pour les performances

#include "lib_bmp.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

typedef struct {
  uint8_t R, G, B;
} type_pixels;

// Division par 9 : floor(x/9) = (x * 116509) >> 20 pour x in [0, 2295]
#define DIV9(x) ((uint8_t)(((uint32_t)(x) * 116509u) >> 20))

// Calcule moy = moyenne et tmp = somme des carres des ecarts pour 9 pixels
// - uint16_t sum : evite le depassement (max 9x255=2295)
// - int16_t di : ecarts signes (evite l'underflow si gi < moy)
// - uint32_t tmp : max 9x255^2=584325, depasse uint16_t
// - pas de /9 sur tmp : A/9 < B/9 <=> A < B, facteur commun
#define SCALAIRE_NAGAO(g0, g1, g2, g3, g4, g5, g6, g7, g8)                     \
  {                                                                            \
    uint16_t sum = (uint16_t)(g0) + (g1) + (g2) + (g3) + (g4) + (g5) + (g6) +  \
                   (g7) + (g8);                                                \
    moy = DIV9(sum);                                                           \
    int16_t d0 = (int16_t)(g0) - moy, d1 = (int16_t)(g1) - moy,                \
            d2 = (int16_t)(g2) - moy, d3 = (int16_t)(g3) - moy,                \
            d4 = (int16_t)(g4) - moy, d5 = (int16_t)(g5) - moy,                \
            d6 = (int16_t)(g6) - moy, d7 = (int16_t)(g7) - moy,                \
            d8 = (int16_t)(g8) - moy;                                          \
    tmp = (uint32_t)(d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 +         \
                     d5 * d5 + d6 * d6 + d7 * d7 + d8 * d8);                   \
  }

// Version scalaire optimisee du filtre de Nagao
static void Nagao_scalaire(uint8_t *restrict src_rgb, uint8_t *restrict dst_rgb,
                           const int Width, const int Height) {
  // nagao, pixels : partages en lecture seule et ecriture sur zones distinctes
  type_pixels pixel_noir = {.R = 0, .G = 0, .B = 0};
  type_pixels *nagao = (type_pixels *)dst_rgb;
  type_pixels *pixels = (type_pixels *)src_rgb;

  // Bords : 2 lignes haut/bas et 2 colonnes gauche/droite en noir
  for (int l0 = 0, l1 = Width, l2 = (Height - 1) * Width,
           l3 = (Height - 2) * Width;
       l0 < Width; l0++, l1++, l2++, l3++)
    nagao[l0] = nagao[l1] = nagao[l2] = nagao[l3] = pixel_noir;

  for (int i = 1, cg = Width,
           cd = (Width << 1) -
                1; // i=1 : ligne 0 deja couverte par la boucle lignes ci-dessus
       i < (Height - 1); i++, cg += Width, cd += Width)
    nagao[cg] = nagao[cd] = nagao[cg + 1] = nagao[cd - 1] = pixel_noir;

// Filtre Nagao : boucle sur les lignes parallelisable (charge egale par ligne)
// schedule(static) : toutes les lignes ont exactement le meme nombre
// d'operations
#pragma omp parallel for schedule(static)
  for (int i = 2; i < (Height - 2); i++) {
    // Offsets recalcules depuis i (requis par la forme canonique OpenMP)
    int il0 = (i - 2) * Width, il1 = (i - 1) * Width, il2 = i * Width,
        il3 = (i + 1) * Width, il4 = (i + 2) * Width;

    // Variables locales a chaque thread (pas de clause private necessaire)
    uint8_t moy, min;
    uint32_t var, tmp;
    uint8_t nagao_00, nagao_01, nagao_02, nagao_03, nagao_04, nagao_10,
        nagao_11, nagao_12, nagao_13, nagao_14, nagao_20, nagao_21, nagao_22,
        nagao_23, nagao_24, nagao_30, nagao_31, nagao_32, nagao_33, nagao_34,
        nagao_40, nagao_41, nagao_42, nagao_43, nagao_44;

    nagao_00 = pixels[il0].R;
    nagao_01 = pixels[il0 + 1].R;
    nagao_02 = pixels[il0 + 2].R;
    nagao_03 = pixels[il0 + 3].R;
    nagao_10 = pixels[il1].R;
    nagao_11 = pixels[il1 + 1].R;
    nagao_12 = pixels[il1 + 2].R;
    nagao_13 = pixels[il1 + 3].R;
    nagao_20 = pixels[il2].R;
    nagao_21 = pixels[il2 + 1].R;
    nagao_22 = pixels[il2 + 2].R;
    nagao_23 = pixels[il2 + 3].R;
    nagao_30 = pixels[il3].R;
    nagao_31 = pixels[il3 + 1].R;
    nagao_32 = pixels[il3 + 2].R;
    nagao_33 = pixels[il3 + 3].R;
    nagao_40 = pixels[il4].R;
    nagao_41 = pixels[il4 + 1].R;
    nagao_42 = pixels[il4 + 2].R;
    nagao_43 = pixels[il4 + 3].R;

    for (int ic = 2, id = 4; ic < (Width - 2); ic++, id++) {
      nagao_04 = pixels[il0 + id].R;
      nagao_14 = pixels[il1 + id].R;
      nagao_24 = pixels[il2 + id].R;
      nagao_34 = pixels[il3 + id].R;
      nagao_44 = pixels[il4 + id].R;

      SCALAIRE_NAGAO(nagao_00, nagao_01, nagao_02, nagao_10, nagao_11, nagao_12,
                     nagao_20, nagao_21, nagao_22); // coin sup gauche
      var = tmp;
      min = moy;

      SCALAIRE_NAGAO(nagao_02, nagao_03, nagao_04, nagao_12, nagao_13, nagao_14,
                     nagao_22, nagao_23, nagao_24); // coin sup droit
      if (tmp < var) {
        var = tmp;
        min = moy;
      }

      SCALAIRE_NAGAO(nagao_20, nagao_21, nagao_22, nagao_30, nagao_31, nagao_32,
                     nagao_40, nagao_41, nagao_42); // coin inf gauche
      if (tmp < var) {
        var = tmp;
        min = moy;
      }

      SCALAIRE_NAGAO(nagao_22, nagao_23, nagao_24, nagao_32, nagao_33, nagao_34,
                     nagao_42, nagao_43, nagao_44); // coin inf droit
      if (tmp < var) {
        var = tmp;
        min = moy;
      }

      SCALAIRE_NAGAO(nagao_01, nagao_02, nagao_03, nagao_11, nagao_12, nagao_13,
                     nagao_21, nagao_22, nagao_23); // ligne sup
      if (tmp < var) {
        var = tmp;
        min = moy;
      }

      SCALAIRE_NAGAO(nagao_21, nagao_22, nagao_23, nagao_31, nagao_32, nagao_33,
                     nagao_41, nagao_42, nagao_43); // ligne inf
      if (tmp < var) {
        var = tmp;
        min = moy;
      }

      SCALAIRE_NAGAO(nagao_10, nagao_11, nagao_12, nagao_20, nagao_21, nagao_22,
                     nagao_30, nagao_31, nagao_32); // col gauche
      if (tmp < var) {
        var = tmp;
        min = moy;
      }

      SCALAIRE_NAGAO(nagao_12, nagao_13, nagao_14, nagao_22, nagao_23, nagao_24,
                     nagao_32, nagao_33, nagao_34); // col droite
      if (tmp < var) {
        var = tmp;
        min = moy;
      }

      SCALAIRE_NAGAO(nagao_11, nagao_12, nagao_13, nagao_21, nagao_22, nagao_23,
                     nagao_31, nagao_32, nagao_33); // centre
      if (tmp < var) {
        var = tmp;
        min = moy;
      }

      nagao[il2 + ic].R = nagao[il2 + ic].G = nagao[il2 + ic].B = min;

      // Rotation des registres (fenetre glissante)
      nagao_00 = nagao_01;
      nagao_01 = nagao_02;
      nagao_02 = nagao_03;
      nagao_03 = nagao_04;
      nagao_10 = nagao_11;
      nagao_11 = nagao_12;
      nagao_12 = nagao_13;
      nagao_13 = nagao_14;
      nagao_20 = nagao_21;
      nagao_21 = nagao_22;
      nagao_22 = nagao_23;
      nagao_23 = nagao_24;
      nagao_30 = nagao_31;
      nagao_31 = nagao_32;
      nagao_32 = nagao_33;
      nagao_33 = nagao_34;
      nagao_40 = nagao_41;
      nagao_41 = nagao_42;
      nagao_42 = nagao_43;
      nagao_43 = nagao_44;
    }
  }
}

// Version NEON du filtre de Nagao (ARM uniquement)
#ifdef __ARM_NEON
enum { PLANE_PAD_RIGHT = 4 };

typedef struct {
  int width;
  int height;
  int stride;
  size_t plane_bytes;
  size_t gray_bytes;
  uint8_t *src_plane;
  uint8_t *dst_gray;
} type_planar_workspace;

static type_planar_workspace g_workspace = {0};

static int ensure_workspace(int Width, int Height) {
  int stride = Width + PLANE_PAD_RIGHT;
  size_t plane_bytes = (size_t)stride * (size_t)Height;
  size_t gray_bytes = (size_t)Width * (size_t)Height;

  if (g_workspace.width != Width || g_workspace.height != Height) {
    free(g_workspace.src_plane);
    free(g_workspace.dst_gray);
    g_workspace.src_plane = (uint8_t *)malloc(plane_bytes);
    g_workspace.dst_gray = (uint8_t *)malloc(gray_bytes);
    if (!g_workspace.src_plane || !g_workspace.dst_gray) {
      free(g_workspace.src_plane);
      free(g_workspace.dst_gray);
      memset(&g_workspace, 0, sizeof(g_workspace));
      return -1;
    }
    g_workspace.width = Width;
    g_workspace.height = Height;
    g_workspace.stride = stride;
    g_workspace.plane_bytes = plane_bytes;
    g_workspace.gray_bytes = gray_bytes;
  }

  memset(g_workspace.src_plane, 0, g_workspace.plane_bytes);
  memset(g_workspace.dst_gray, 0, g_workspace.gray_bytes);
  return 0;
}

static void release_workspace(void) {
  free(g_workspace.src_plane);
  free(g_workspace.dst_gray);
  memset(&g_workspace, 0, sizeof(g_workspace));
}

static void extract_work_channel_plane(const uint8_t *src_rgb, uint8_t *plane,
                                       int Width, int Height, int stride) {
  const type_pixels *src = (const type_pixels *)src_rgb;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < Height; i++) {
    int src_row = i * Width;
    int dst_row = i * stride;

    for (int j = 0; j < Width; j++)
      plane[dst_row + j] = src[src_row + j].R;
  }
}

static void expand_gray_to_rgb(const uint8_t *gray, uint8_t *dst_rgb, int Width,
                               int Height) {
  type_pixels *dst = (type_pixels *)dst_rgb;
  size_t count = (size_t)Width * (size_t)Height;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < count; i++)
    dst[i].R = dst[i].G = dst[i].B = gray[i];
}

static inline uint8x8_t div9_u16_exact_0_2295(uint16x8_t sum) {
  uint32x4_t lo32 = vmull_n_u16(vget_low_u16(sum), 7282);
  uint32x4_t hi32 = vmull_n_u16(vget_high_u16(sum), 7282);
  return vmovn_u16(vcombine_u16(vshrn_n_u32(lo32, 16), vshrn_n_u32(hi32, 16)));
}

#define ROW_SUM3(out, a, b, c)                                                 \
  do {                                                                         \
    out = vaddw_u8(vmovl_u8(a), b);                                            \
    out = vaddw_u8(out, c);                                                    \
  } while (0)

#define ROW_SUMSQ3(lo, hi, a, b, c)                                            \
  do {                                                                         \
    uint16x8_t sq = vmull_u8(a, a);                                            \
    lo = vmovl_u16(vget_low_u16(sq));                                          \
    hi = vmovl_u16(vget_high_u16(sq));                                         \
    sq = vmull_u8(b, b);                                                       \
    lo = vaddw_u16(lo, vget_low_u16(sq));                                      \
    hi = vaddw_u16(hi, vget_high_u16(sq));                                     \
    sq = vmull_u8(c, c);                                                       \
    lo = vaddw_u16(lo, vget_low_u16(sq));                                      \
    hi = vaddw_u16(hi, vget_high_u16(sq));                                     \
  } while (0)

#define ADD3_U32X8(out_lo, out_hi, a_lo, a_hi, b_lo, b_hi, c_lo, c_hi)        \
  do {                                                                         \
    out_lo = vaddq_u32(vaddq_u32(a_lo, b_lo), c_lo);                           \
    out_hi = vaddq_u32(vaddq_u32(a_hi, b_hi), c_hi);                           \
  } while (0)

#define NEON_COMPUTE_STATS(sum_in, sq_lo_in, sq_hi_in, mean_out, var_lo_out,   \
                           var_hi_out)                                         \
  do {                                                                         \
    uint16x8_t mean16;                                                         \
    uint32x4_t sum_lo, sum_hi;                                                 \
    uint32x4_t mean_lo, mean_hi;                                               \
    uint32x4_t term_lo, term_hi;                                               \
    mean_out = div9_u16_exact_0_2295(sum_in);                                  \
    mean16 = vmovl_u8(mean_out);                                               \
    sum_lo = vmovl_u16(vget_low_u16(sum_in));                                  \
    sum_hi = vmovl_u16(vget_high_u16(sum_in));                                 \
    mean_lo = vmovl_u16(vget_low_u16(mean16));                                 \
    mean_hi = vmovl_u16(vget_high_u16(mean16));                                \
    term_lo = vsubq_u32(vshlq_n_u32(sum_lo, 1), vmulq_n_u32(mean_lo, 9));      \
    term_hi = vsubq_u32(vshlq_n_u32(sum_hi, 1), vmulq_n_u32(mean_hi, 9));      \
    var_lo_out = vsubq_u32(sq_lo_in, vmulq_u32(mean_lo, term_lo));             \
    var_hi_out = vsubq_u32(sq_hi_in, vmulq_u32(mean_hi, term_hi));             \
  } while (0)

#define NEON_SELECT_BEST_U32(var_lo, var_hi, moy_in, best_var_lo, best_var_hi, \
                             best_moy)                                         \
  do {                                                                         \
    uint32x4_t mask_lo = vcltq_u32(var_lo, best_var_lo);                       \
    uint32x4_t mask_hi = vcltq_u32(var_hi, best_var_hi);                       \
    best_var_lo = vbslq_u32(mask_lo, var_lo, best_var_lo);                     \
    best_var_hi = vbslq_u32(mask_hi, var_hi, best_var_hi);                     \
    best_moy = vbsl_u8(vmovn_u16(vcombine_u16(vmovn_u32(mask_lo),              \
                                              vmovn_u32(mask_hi))),            \
                       moy_in, best_moy);                                      \
  } while (0)

static void Nagao_neon_gray_kernel(const uint8_t *src_plane, uint8_t *dst_gray,
                                   int Width, int Height, int stride) {
#pragma omp parallel for schedule(static)
  for (int i = 2; i < (Height - 2); i++) {
    int il0 = (i - 2) * stride, il1 = (i - 1) * stride, il2 = i * stride,
        il3 = (i + 1) * stride, il4 = (i + 2) * stride;
    int out_row = i * Width;
    int ic;

    for (ic = 2; ic <= (Width - 10); ic += 8) {
      uint8x8_t b0 = vld1_u8(src_plane + il0 + ic - 2);
      uint8x8_t n0 = vld1_u8(src_plane + il0 + ic + 6);
      uint8x8_t v_00 = b0;
      uint8x8_t v_01 = vext_u8(b0, n0, 1);
      uint8x8_t v_02 = vext_u8(b0, n0, 2);
      uint8x8_t v_03 = vext_u8(b0, n0, 3);
      uint8x8_t v_04 = vext_u8(b0, n0, 4);

      uint8x8_t b1 = vld1_u8(src_plane + il1 + ic - 2);
      uint8x8_t n1 = vld1_u8(src_plane + il1 + ic + 6);
      uint8x8_t v_10 = b1;
      uint8x8_t v_11 = vext_u8(b1, n1, 1);
      uint8x8_t v_12 = vext_u8(b1, n1, 2);
      uint8x8_t v_13 = vext_u8(b1, n1, 3);
      uint8x8_t v_14 = vext_u8(b1, n1, 4);

      uint8x8_t b2 = vld1_u8(src_plane + il2 + ic - 2);
      uint8x8_t n2 = vld1_u8(src_plane + il2 + ic + 6);
      uint8x8_t v_20 = b2;
      uint8x8_t v_21 = vext_u8(b2, n2, 1);
      uint8x8_t v_22 = vext_u8(b2, n2, 2);
      uint8x8_t v_23 = vext_u8(b2, n2, 3);
      uint8x8_t v_24 = vext_u8(b2, n2, 4);

      uint8x8_t b3 = vld1_u8(src_plane + il3 + ic - 2);
      uint8x8_t n3 = vld1_u8(src_plane + il3 + ic + 6);
      uint8x8_t v_30 = b3;
      uint8x8_t v_31 = vext_u8(b3, n3, 1);
      uint8x8_t v_32 = vext_u8(b3, n3, 2);
      uint8x8_t v_33 = vext_u8(b3, n3, 3);
      uint8x8_t v_34 = vext_u8(b3, n3, 4);

      uint8x8_t b4 = vld1_u8(src_plane + il4 + ic - 2);
      uint8x8_t n4 = vld1_u8(src_plane + il4 + ic + 6);
      uint8x8_t v_40 = b4;
      uint8x8_t v_41 = vext_u8(b4, n4, 1);
      uint8x8_t v_42 = vext_u8(b4, n4, 2);
      uint8x8_t v_43 = vext_u8(b4, n4, 3);
      uint8x8_t v_44 = vext_u8(b4, n4, 4);

      uint16x8_t s00, s01, s02, s10, s11, s12, s20, s21, s22, s30, s31, s32,
          s40, s41, s42;
      uint32x4_t q00_lo, q00_hi, q01_lo, q01_hi, q02_lo, q02_hi;
      uint32x4_t q10_lo, q10_hi, q11_lo, q11_hi, q12_lo, q12_hi;
      uint32x4_t q20_lo, q20_hi, q21_lo, q21_hi, q22_lo, q22_hi;
      uint32x4_t q30_lo, q30_hi, q31_lo, q31_hi, q32_lo, q32_hi;
      uint32x4_t q40_lo, q40_hi, q41_lo, q41_hi, q42_lo, q42_hi;
      uint16x8_t sum_tmp;
      uint32x4_t sq_lo, sq_hi, best_var_lo, best_var_hi, cur_var_lo, cur_var_hi;
      uint8x8_t best_moy, cur_moy;

      ROW_SUM3(s00, v_00, v_01, v_02);
      ROW_SUM3(s01, v_01, v_02, v_03);
      ROW_SUM3(s02, v_02, v_03, v_04);
      ROW_SUM3(s10, v_10, v_11, v_12);
      ROW_SUM3(s11, v_11, v_12, v_13);
      ROW_SUM3(s12, v_12, v_13, v_14);
      ROW_SUM3(s20, v_20, v_21, v_22);
      ROW_SUM3(s21, v_21, v_22, v_23);
      ROW_SUM3(s22, v_22, v_23, v_24);
      ROW_SUM3(s30, v_30, v_31, v_32);
      ROW_SUM3(s31, v_31, v_32, v_33);
      ROW_SUM3(s32, v_32, v_33, v_34);
      ROW_SUM3(s40, v_40, v_41, v_42);
      ROW_SUM3(s41, v_41, v_42, v_43);
      ROW_SUM3(s42, v_42, v_43, v_44);

      ROW_SUMSQ3(q00_lo, q00_hi, v_00, v_01, v_02);
      ROW_SUMSQ3(q01_lo, q01_hi, v_01, v_02, v_03);
      ROW_SUMSQ3(q02_lo, q02_hi, v_02, v_03, v_04);
      ROW_SUMSQ3(q10_lo, q10_hi, v_10, v_11, v_12);
      ROW_SUMSQ3(q11_lo, q11_hi, v_11, v_12, v_13);
      ROW_SUMSQ3(q12_lo, q12_hi, v_12, v_13, v_14);
      ROW_SUMSQ3(q20_lo, q20_hi, v_20, v_21, v_22);
      ROW_SUMSQ3(q21_lo, q21_hi, v_21, v_22, v_23);
      ROW_SUMSQ3(q22_lo, q22_hi, v_22, v_23, v_24);
      ROW_SUMSQ3(q30_lo, q30_hi, v_30, v_31, v_32);
      ROW_SUMSQ3(q31_lo, q31_hi, v_31, v_32, v_33);
      ROW_SUMSQ3(q32_lo, q32_hi, v_32, v_33, v_34);
      ROW_SUMSQ3(q40_lo, q40_hi, v_40, v_41, v_42);
      ROW_SUMSQ3(q41_lo, q41_hi, v_41, v_42, v_43);
      ROW_SUMSQ3(q42_lo, q42_hi, v_42, v_43, v_44);

      sum_tmp = vaddq_u16(vaddq_u16(s00, s10), s20);
      ADD3_U32X8(sq_lo, sq_hi, q00_lo, q00_hi, q10_lo, q10_hi, q20_lo, q20_hi);
      NEON_COMPUTE_STATS(sum_tmp, sq_lo, sq_hi, best_moy, best_var_lo,
                         best_var_hi);

      sum_tmp = vaddq_u16(vaddq_u16(s02, s12), s22);
      ADD3_U32X8(sq_lo, sq_hi, q02_lo, q02_hi, q12_lo, q12_hi, q22_lo, q22_hi);
      NEON_COMPUTE_STATS(sum_tmp, sq_lo, sq_hi, cur_moy, cur_var_lo, cur_var_hi);
      NEON_SELECT_BEST_U32(cur_var_lo, cur_var_hi, cur_moy, best_var_lo,
                           best_var_hi, best_moy);

      sum_tmp = vaddq_u16(vaddq_u16(s20, s30), s40);
      ADD3_U32X8(sq_lo, sq_hi, q20_lo, q20_hi, q30_lo, q30_hi, q40_lo, q40_hi);
      NEON_COMPUTE_STATS(sum_tmp, sq_lo, sq_hi, cur_moy, cur_var_lo, cur_var_hi);
      NEON_SELECT_BEST_U32(cur_var_lo, cur_var_hi, cur_moy, best_var_lo,
                           best_var_hi, best_moy);

      sum_tmp = vaddq_u16(vaddq_u16(s22, s32), s42);
      ADD3_U32X8(sq_lo, sq_hi, q22_lo, q22_hi, q32_lo, q32_hi, q42_lo, q42_hi);
      NEON_COMPUTE_STATS(sum_tmp, sq_lo, sq_hi, cur_moy, cur_var_lo, cur_var_hi);
      NEON_SELECT_BEST_U32(cur_var_lo, cur_var_hi, cur_moy, best_var_lo,
                           best_var_hi, best_moy);

      sum_tmp = vaddq_u16(vaddq_u16(s01, s11), s21);
      ADD3_U32X8(sq_lo, sq_hi, q01_lo, q01_hi, q11_lo, q11_hi, q21_lo, q21_hi);
      NEON_COMPUTE_STATS(sum_tmp, sq_lo, sq_hi, cur_moy, cur_var_lo, cur_var_hi);
      NEON_SELECT_BEST_U32(cur_var_lo, cur_var_hi, cur_moy, best_var_lo,
                           best_var_hi, best_moy);

      sum_tmp = vaddq_u16(vaddq_u16(s21, s31), s41);
      ADD3_U32X8(sq_lo, sq_hi, q21_lo, q21_hi, q31_lo, q31_hi, q41_lo, q41_hi);
      NEON_COMPUTE_STATS(sum_tmp, sq_lo, sq_hi, cur_moy, cur_var_lo, cur_var_hi);
      NEON_SELECT_BEST_U32(cur_var_lo, cur_var_hi, cur_moy, best_var_lo,
                           best_var_hi, best_moy);

      sum_tmp = vaddq_u16(vaddq_u16(s10, s20), s30);
      ADD3_U32X8(sq_lo, sq_hi, q10_lo, q10_hi, q20_lo, q20_hi, q30_lo, q30_hi);
      NEON_COMPUTE_STATS(sum_tmp, sq_lo, sq_hi, cur_moy, cur_var_lo, cur_var_hi);
      NEON_SELECT_BEST_U32(cur_var_lo, cur_var_hi, cur_moy, best_var_lo,
                           best_var_hi, best_moy);

      sum_tmp = vaddq_u16(vaddq_u16(s12, s22), s32);
      ADD3_U32X8(sq_lo, sq_hi, q12_lo, q12_hi, q22_lo, q22_hi, q32_lo, q32_hi);
      NEON_COMPUTE_STATS(sum_tmp, sq_lo, sq_hi, cur_moy, cur_var_lo, cur_var_hi);
      NEON_SELECT_BEST_U32(cur_var_lo, cur_var_hi, cur_moy, best_var_lo,
                           best_var_hi, best_moy);

      sum_tmp = vaddq_u16(vaddq_u16(s11, s21), s31);
      ADD3_U32X8(sq_lo, sq_hi, q11_lo, q11_hi, q21_lo, q21_hi, q31_lo, q31_hi);
      NEON_COMPUTE_STATS(sum_tmp, sq_lo, sq_hi, cur_moy, cur_var_lo, cur_var_hi);
      NEON_SELECT_BEST_U32(cur_var_lo, cur_var_hi, cur_moy, best_var_lo,
                           best_var_hi, best_moy);

      vst1_u8(dst_gray + out_row + ic, best_moy);
    }

    if (ic < (Width - 2)) {
      uint8_t nagao_00, nagao_01, nagao_02, nagao_03, nagao_04, nagao_10,
          nagao_11, nagao_12, nagao_13, nagao_14, nagao_20, nagao_21, nagao_22,
          nagao_23, nagao_24, nagao_30, nagao_31, nagao_32, nagao_33, nagao_34,
          nagao_40, nagao_41, nagao_42, nagao_43, nagao_44;
      uint8_t moy, min;
      uint32_t var, tmp;

      nagao_00 = src_plane[il0 + ic - 2];
      nagao_01 = src_plane[il0 + ic - 1];
      nagao_02 = src_plane[il0 + ic];
      nagao_03 = src_plane[il0 + ic + 1];
      nagao_10 = src_plane[il1 + ic - 2];
      nagao_11 = src_plane[il1 + ic - 1];
      nagao_12 = src_plane[il1 + ic];
      nagao_13 = src_plane[il1 + ic + 1];
      nagao_20 = src_plane[il2 + ic - 2];
      nagao_21 = src_plane[il2 + ic - 1];
      nagao_22 = src_plane[il2 + ic];
      nagao_23 = src_plane[il2 + ic + 1];
      nagao_30 = src_plane[il3 + ic - 2];
      nagao_31 = src_plane[il3 + ic - 1];
      nagao_32 = src_plane[il3 + ic];
      nagao_33 = src_plane[il3 + ic + 1];
      nagao_40 = src_plane[il4 + ic - 2];
      nagao_41 = src_plane[il4 + ic - 1];
      nagao_42 = src_plane[il4 + ic];
      nagao_43 = src_plane[il4 + ic + 1];

      for (int id = ic + 2; ic < (Width - 2); ic++, id++) {
        nagao_04 = src_plane[il0 + id];
        nagao_14 = src_plane[il1 + id];
        nagao_24 = src_plane[il2 + id];
        nagao_34 = src_plane[il3 + id];
        nagao_44 = src_plane[il4 + id];

        SCALAIRE_NAGAO(nagao_00, nagao_01, nagao_02, nagao_10, nagao_11,
                       nagao_12, nagao_20, nagao_21, nagao_22);
        var = tmp;
        min = moy;

        SCALAIRE_NAGAO(nagao_02, nagao_03, nagao_04, nagao_12, nagao_13,
                       nagao_14, nagao_22, nagao_23, nagao_24);
        if (tmp < var) {
          var = tmp;
          min = moy;
        }

        SCALAIRE_NAGAO(nagao_20, nagao_21, nagao_22, nagao_30, nagao_31,
                       nagao_32, nagao_40, nagao_41, nagao_42);
        if (tmp < var) {
          var = tmp;
          min = moy;
        }

        SCALAIRE_NAGAO(nagao_22, nagao_23, nagao_24, nagao_32, nagao_33,
                       nagao_34, nagao_42, nagao_43, nagao_44);
        if (tmp < var) {
          var = tmp;
          min = moy;
        }

        SCALAIRE_NAGAO(nagao_01, nagao_02, nagao_03, nagao_11, nagao_12,
                       nagao_13, nagao_21, nagao_22, nagao_23);
        if (tmp < var) {
          var = tmp;
          min = moy;
        }

        SCALAIRE_NAGAO(nagao_21, nagao_22, nagao_23, nagao_31, nagao_32,
                       nagao_33, nagao_41, nagao_42, nagao_43);
        if (tmp < var) {
          var = tmp;
          min = moy;
        }

        SCALAIRE_NAGAO(nagao_10, nagao_11, nagao_12, nagao_20, nagao_21,
                       nagao_22, nagao_30, nagao_31, nagao_32);
        if (tmp < var) {
          var = tmp;
          min = moy;
        }

        SCALAIRE_NAGAO(nagao_12, nagao_13, nagao_14, nagao_22, nagao_23,
                       nagao_24, nagao_32, nagao_33, nagao_34);
        if (tmp < var) {
          var = tmp;
          min = moy;
        }

        SCALAIRE_NAGAO(nagao_11, nagao_12, nagao_13, nagao_21, nagao_22,
                       nagao_23, nagao_31, nagao_32, nagao_33);
        if (tmp < var) {
          var = tmp;
          min = moy;
        }

        dst_gray[out_row + ic] = min;

        nagao_00 = nagao_01;
        nagao_01 = nagao_02;
        nagao_02 = nagao_03;
        nagao_03 = nagao_04;
        nagao_10 = nagao_11;
        nagao_11 = nagao_12;
        nagao_12 = nagao_13;
        nagao_13 = nagao_14;
        nagao_20 = nagao_21;
        nagao_21 = nagao_22;
        nagao_22 = nagao_23;
        nagao_23 = nagao_24;
        nagao_30 = nagao_31;
        nagao_31 = nagao_32;
        nagao_32 = nagao_33;
        nagao_33 = nagao_34;
        nagao_40 = nagao_41;
        nagao_41 = nagao_42;
        nagao_42 = nagao_43;
        nagao_43 = nagao_44;
      }
    }
  }
}

static void Nagao_neon(uint8_t *restrict src_rgb, uint8_t *restrict dst_rgb,
                       const int Width, const int Height) {
  if (ensure_workspace(Width, Height)) {
    Nagao_scalaire(src_rgb, dst_rgb, Width, Height);
    return;
  }

  extract_work_channel_plane(src_rgb, g_workspace.src_plane, Width, Height,
                             g_workspace.stride);
  Nagao_neon_gray_kernel(g_workspace.src_plane, g_workspace.dst_gray, Width,
                         Height, g_workspace.stride);
  expand_gray_to_rgb(g_workspace.dst_gray, dst_rgb, Width, Height);
}
#endif

// pour la mesure (mediane sur NB_RUNS runs)
#define NB_RUNS 10

static int cmp_double(const void *a, const void *b) {
  double da = *(const double *)a, db = *(const double *)b;
  return (da > db) - (da < db);
}
static double mediane(double *t, int n) {
  qsort(t, n, sizeof(double), cmp_double);
  return t[n / 2];
}
static double mesurer(void (*fn)(uint8_t *, uint8_t *, int, int), uint8_t *src,
                      uint8_t *dst, int W, int H, int nb_threads) {
  double t[NB_RUNS];
  omp_set_num_threads(nb_threads);
  for (int r = 0; r < NB_RUNS; r++) {
    double t0 = omp_get_wtime();
    fn(src, dst, W, H);
    t[r] = (omp_get_wtime() - t0) * 1000.0;
  }
  return mediane(t, NB_RUNS);
}

int main(void) {
  type_bitmap bmp_src, bmp_dst;

  if (bmp_open("caterham_gris_bruit.bmp", &bmp_src)) {
    printf("Erreur : impossible d'ouvrir l'image source\n");
    return 1;
  }

  memcpy(&(bmp_dst.file_header), &(bmp_src.file_header), 54);
  bmp_dst.pixels = (uint8_t *)malloc(bmp_src.picture_header.height *
                                     bmp_src.picture_header.width * 3);
  if (!bmp_dst.pixels) {
    printf("Erreur allocation\n");
    free(bmp_src.pixels);
    return 1;
  }

  int W = (int)bmp_src.picture_header.width;
  int H = (int)bmp_src.picture_header.height;
  int nb_threads = omp_get_max_threads();

#ifdef MODE_PHOTO
  Nagao_scalaire(bmp_src.pixels, bmp_dst.pixels, W, H);
  bmp_write("caterham_nagao.bmp", bmp_dst);
  printf("Image scalaire : caterham_nagao.bmp\n");

#ifdef __ARM_NEON
  Nagao_neon(bmp_src.pixels, bmp_dst.pixels, W, H);
  bmp_write("caterham_nagao_simd.bmp", bmp_dst);
  printf("Image NEON : caterham_nagao_simd.bmp\n");
#endif

#elif defined(MODE_PERF)
  double sc_1t =
      mesurer(Nagao_scalaire, bmp_src.pixels, bmp_dst.pixels, W, H, 1);
  double sc_nt =
      mesurer(Nagao_scalaire, bmp_src.pixels, bmp_dst.pixels, W, H, nb_threads);

#ifdef __ARM_NEON
  double neon_1t = mesurer(Nagao_neon, bmp_src.pixels, bmp_dst.pixels, W, H, 1);
  double neon_nt =
      mesurer(Nagao_neon, bmp_src.pixels, bmp_dst.pixels, W, H, nb_threads);
#endif

  printf("\n=== Nagao — Performances (mediane %d runs, %d threads) ===\n\n",
         NB_RUNS, nb_threads);
  printf(" %-10s | %11s | %13s | %11s\n", "Version", "1 thread", "N threads",
         "Speedup OMP");
  printf(" -----------|-------------|---------------|------------\n");
  printf(" %-10s | %8.3f ms | %10.3f ms | %9.2fx\n", "Scalaire", sc_1t, sc_nt,
         sc_1t / sc_nt);
#ifdef __ARM_NEON
  printf(" %-10s | %8.3f ms | %10.3f ms | %9.2fx\n", "NEON", neon_1t, neon_nt,
         neon_1t / neon_nt);
  printf(" -----------|-------------|---------------|------------\n");
  printf(" %-10s | %11.2fx | %13.2fx |\n", "Speedup NEON", sc_1t / neon_1t,
         sc_nt / neon_nt);
#endif
  printf("\n");

#else
  printf("Decommenter MODE_PHOTO ou MODE_PERF\n");
#endif

#ifdef __ARM_NEON
  release_workspace();
#endif
  free(bmp_src.pixels);
  free(bmp_dst.pixels);
  return 0;
}
