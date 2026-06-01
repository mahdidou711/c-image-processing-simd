#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include <omp.h>
#include "lib_bmp.h"

#define SIMD_VERSION

#define N               4
#define BLACK           0x00
#define WHITE           0xFF
#define NB_IMAGES       200
#define IMAGE_NAME      "Image"
#define PATH_SRC        "./images_src/"
#ifdef  SIMD_VERSION
#define IMAGE_SD        "Image_simd_"
#define IMAGE_SD_E      "Image_simd_e_"
#define IMAGE_SD_E_D    "Image_simd_e_d_"
#define PATH_SD         "./images_simd/"
#else
#define IMAGE_SD        "Image_sca_"
#define IMAGE_SD_E      "Image_sca_e_"
#define IMAGE_SD_E_D    "Image_sca_e_d_"
#define PATH_SD         "./images_scalaire/"
#endif

uint64_t system_nanoTime() {
        struct timespec now;
        clock_gettime(0, &now);
        return now.tv_sec * 1000000000LL + now.tv_nsec;
    }

void SD_Initialization(int32_t nb_pixels, uint8_t *pixels_src, uint8_t *moy, uint16_t* var, uint8_t *pixels_sd)
{
#ifndef  SIMD_VERSION
    uint32_t i;
    for(i = 0; i < nb_pixels; i++)
    {
        moy[i] = pixels_src[i];
        var[i] = 0;
        pixels_sd[i] = BLACK;
    }
#else
    uint32_t i;
    uint8x16_t v_src;
    uint8x16_t v_black = vdupq_n_u8(BLACK);
    uint16x8_t v_zero16 = vdupq_n_u16(0);

    #pragma omp parallel for schedule(static) private(v_src)
    for (i = 0; i < nb_pixels; i += 16)
    {
        v_src = vld1q_u8(pixels_src + i);
        vst1q_u8(moy + i, v_src);
        vst1q_u16(var + i,     v_zero16);
        vst1q_u16(var + i + 8, v_zero16);
        vst1q_u8(pixels_sd + i, v_black);
    }
#endif
}

void Sigma_Delta(int32_t nb_pixels, uint8_t *pixels_in, uint8_t *moy, uint16_t* var, uint8_t *pixels_out)
{
#ifndef  SIMD_VERSION
    uint32_t i;
    uint8_t tmp_moy, tmp_pixel, delta;
    uint16_t tmp_var, delta_N;

    for(i = 0; i < nb_pixels; i++)
    {
        tmp_moy = moy[i];
        tmp_pixel = pixels_in[i];
        tmp_var = var[i];

        if (tmp_moy < tmp_pixel) tmp_moy++;
        if (tmp_moy > tmp_pixel) tmp_moy--;

        delta = abs(tmp_moy - tmp_pixel);
        delta_N = delta * N;

        if (delta != 0)
        {
            if (tmp_var < delta_N) tmp_var++;
            if (tmp_var > delta_N) tmp_var--;
        }

        if (delta < tmp_var)
            tmp_pixel = WHITE;
        else
            tmp_pixel = BLACK;

        moy[i] = tmp_moy;
        pixels_out[i] = tmp_pixel;
        var[i] = tmp_var;
    }
#else
    uint32_t i;
    
uint8x16_t v_zero8 = vdupq_n_u8(0);

    #pragma omp parallel for schedule(static)
    for (i = 0; i < nb_pixels; i += 32)
    {
        uint8_t  *pA_in  = pixels_in  + i;
        uint8_t  *pA_moy = moy        + i;
        uint16_t *pA_var = var        + i;
        uint8_t  *pA_out = pixels_out + i;
        uint8_t  *pB_in  = pixels_in  + i + 16;
        uint8_t  *pB_moy = moy        + i + 16;
        uint16_t *pB_var = var        + i + 16;
        uint8_t  *pB_out = pixels_out + i + 16;

        // --- Chargements A et B interleaves ---
        uint8x16_t vA_pixel = vld1q_u8(pA_in);
        uint8x16_t vB_pixel = vld1q_u8(pB_in);
        uint8x16_t vA_moy   = vld1q_u8(pA_moy);
        uint8x16_t vB_moy   = vld1q_u8(pB_moy);

        // var : u16 en memoire, octet bas = valeur utile (<= 255), octet haut = 0
        uint8x16x2_t vpA = vld2q_u8((uint8_t*)pA_var);
        uint8x16x2_t vpB = vld2q_u8((uint8_t*)pB_var);
        uint8x16_t vA_var8 = vpA.val[0];
        uint8x16_t vB_var8 = vpB.val[0];

        // --- Mise a jour moyenne (mask-as-number u8) ---
        uint8x16_t mA_lt = vcltq_u8(vA_moy, vA_pixel);
        uint8x16_t mB_lt = vcltq_u8(vB_moy, vB_pixel);
        uint8x16_t mA_gt = vcgtq_u8(vA_moy, vA_pixel);
        uint8x16_t mB_gt = vcgtq_u8(vB_moy, vB_pixel);
        vA_moy = vsubq_u8(vA_moy, mA_lt);
        vB_moy = vsubq_u8(vB_moy, mB_lt);
        vA_moy = vaddq_u8(vA_moy, mA_gt);
        vB_moy = vaddq_u8(vB_moy, mB_gt);

        // --- Delta = |moy - pixel| ---
        uint8x16_t vA_delta = vsubq_u8(vmaxq_u8(vA_moy, vA_pixel),
                                        vminq_u8(vA_moy, vA_pixel));
        uint8x16_t vB_delta = vsubq_u8(vmaxq_u8(vB_moy, vB_pixel),
                                        vminq_u8(vB_moy, vB_pixel));

        // --- deltaN = min(4*delta, 255) via 2 vqaddq_u8 successifs ---
        uint8x16_t vA_2d     = vqaddq_u8(vA_delta, vA_delta);
        uint8x16_t vB_2d     = vqaddq_u8(vB_delta, vB_delta);
        uint8x16_t vA_deltaN8 = vqaddq_u8(vA_2d, vA_2d);
        uint8x16_t vB_deltaN8 = vqaddq_u8(vB_2d, vB_2d);

        // --- Masque delta != 0 ---
        uint8x16_t mask_nzA = vtstq_u8(vA_delta, vA_delta);
        uint8x16_t mask_nzB = vtstq_u8(vB_delta, vB_delta);

        // --- Mise a jour var en u8 (mask-as-number u8) ---
        uint8x16_t mlt_vA = vcltq_u8(vA_var8, vA_deltaN8);
        uint8x16_t mlt_vB = vcltq_u8(vB_var8, vB_deltaN8);
        uint8x16_t mgt_vA = vandq_u8(vcgtq_u8(vA_var8, vA_deltaN8), mask_nzA);
        uint8x16_t mgt_vB = vandq_u8(vcgtq_u8(vB_var8, vB_deltaN8), mask_nzB);
        vA_var8 = vsubq_u8(vA_var8, mlt_vA);
        vB_var8 = vsubq_u8(vB_var8, mlt_vB);
        vA_var8 = vaddq_u8(vA_var8, mgt_vA);
        vB_var8 = vaddq_u8(vB_var8, mgt_vB);

        // --- Classification directe u8 : delta < var => WHITE (0xFF) ---
        uint8x16_t vA_out = vcltq_u8(vA_delta, vA_var8);
        uint8x16_t vB_out = vcltq_u8(vB_delta, vB_var8);

        // --- Stores ---
        vst1q_u8(pA_moy, vA_moy);
        vst1q_u8(pB_moy, vB_moy);
        // var : octet bas = valeur, octet haut = 0
        uint8x16x2_t vsA;
        vsA.val[0] = vA_var8;
        vsA.val[1] = v_zero8;
        uint8x16x2_t vsB;
        vsB.val[0] = vB_var8;
        vsB.val[1] = v_zero8;
        vst2q_u8((uint8_t*)pA_var, vsA);
        vst2q_u8((uint8_t*)pB_var, vsB);
        vst1q_u8(pA_out, vA_out);
        vst1q_u8(pB_out, vB_out);
    }
#endif
}

void Morpho_Initialization(uint32_t Width, uint32_t Height, uint8_t *pixels)
{
#ifndef  SIMD_VERSION
    uint32_t i;

    for(i = 0; i < Width; i++)
        pixels[i] = pixels[i + (Height-1)*Width] = WHITE;
    for(i = 1; i < (Height-1); i++)
        pixels[i * Width] = pixels[(i + 1) * Width - 1] = WHITE;
#else
    uint32_t i;
    uint8x16_t v_white = vdupq_n_u8(WHITE);

    uint8_t *ptr_top = pixels;
    uint8_t *ptr_bot = pixels + (Height - 1) * Width;
    #pragma omp parallel for schedule(static)
    for (i = 0; i < Width; i += 16)
    {
        vst1q_u8(ptr_top + i, v_white);
        vst1q_u8(ptr_bot + i, v_white);
    }

    // adressage par offsets : ptr += Width au lieu de i*Width
    // hoisting : right_off = Width-16 calcule une seule fois hors boucle
    uint32_t right_off = Width - 16;
    uint8_t *ptr_row   = pixels + Width;
    for (i = 1; i < (Height - 1); i++)
    {
        vst1q_u8(ptr_row,             v_white);
        vst1q_u8(ptr_row + right_off, v_white);
        ptr_row += Width;
    }
#endif
}

void Erosion(uint32_t Width, uint32_t Height, uint8_t *pixels_in, uint8_t *pixels_out)
{
#ifndef  SIMD_VERSION
    uint32_t i, j;

    for(i = 1; i < (Height-1); i++)
        for(j = 1; j < (Width-1); j++)
            if( (pixels_in[(i-1)*Width+j-1] == BLACK) &&
                (pixels_in[(i-1)*Width+j  ] == BLACK) &&
                (pixels_in[(i-1)*Width+j+1] == BLACK) &&
                (pixels_in[ i   *Width+j-1] == BLACK) &&
                (pixels_in[ i   *Width+j  ] == BLACK) &&
                (pixels_in[ i   *Width+j+1] == BLACK) &&
                (pixels_in[(i+1)*Width+j-1] == BLACK) &&
                (pixels_in[(i+1)*Width+j  ] == BLACK) &&
                (pixels_in[(i+1)*Width+j+1] == BLACK) )
                pixels_out[ i   *Width+j  ] = BLACK;
            else
                pixels_out[ i   *Width+j  ] = WHITE;
#else
    uint32_t i, j;
    uint8x16_t v_white = vdupq_n_u8(WHITE);

    for (j = 0; j < Width; j += 16)
    {
        vst1q_u8(pixels_out + j, v_white);
        vst1q_u8(pixels_out + (Height - 1) * Width + j, v_white);
    }

    // hoisting : borne et offset de bord calcules une seule fois
    uint32_t jlim    = Width - 16;
    uint32_t right16 = Width - 16;

    #pragma omp parallel for schedule(static) private(j)
    for (i = 1; i < (Height - 1); i++)
    {
        uint8_t *row_prev = pixels_in  + (i - 1) * Width;
        uint8_t *row_curr = pixels_in  +  i      * Width;
        uint8_t *row_next = pixels_in  + (i + 1) * Width;
        uint8_t *row_out  = pixels_out +  i      * Width;

        vst1q_u8(row_out,             v_white);
        vst1q_u8(row_out + right16,   v_white);

        for (j = 16; j < jlim; j += 16)
        {
            uint8x16_t v_p0 = vld1q_u8(row_prev + j - 1);
            uint8x16_t v_p1 = vld1q_u8(row_prev + j);
            uint8x16_t v_p2 = vld1q_u8(row_prev + j + 1);

            uint8x16_t v_c0 = vld1q_u8(row_curr + j - 1);
            uint8x16_t v_c1 = vld1q_u8(row_curr + j);
            uint8x16_t v_c2 = vld1q_u8(row_curr + j + 1);

            uint8x16_t v_n0 = vld1q_u8(row_next + j - 1);
            uint8x16_t v_n1 = vld1q_u8(row_next + j);
            uint8x16_t v_n2 = vld1q_u8(row_next + j + 1);

            // arbre OR equilibre : profondeur 4 au lieu de 8
            // niveau 1 : 4 OR independants
            uint8x16_t t0 = vorrq_u8(v_p0, v_p1);
            uint8x16_t t1 = vorrq_u8(v_p2, v_c0);
            uint8x16_t t2 = vorrq_u8(v_c1, v_c2);
            uint8x16_t t3 = vorrq_u8(v_n0, v_n1);
            // niveau 2
            uint8x16_t t4 = vorrq_u8(t0, t1);
            uint8x16_t t5 = vorrq_u8(t2, t3);
            // niveaux 3 et 4
            uint8x16_t v_res = vorrq_u8(vorrq_u8(t4, t5), v_n2);

            vst1q_u8(row_out + j, v_res);
        }
    }
#endif
}

void Dilatation(uint32_t Width, uint32_t Height, uint8_t *pixels_in, uint8_t *pixels_out)
{
#ifndef  SIMD_VERSION
    uint32_t i, j;

    for(i = 1; i < (Height-1); i++)
        for(j = 1; j < (Width-1); j++)
            if( (pixels_in[(i-1)*Width+j-1] == BLACK) ||
                (pixels_in[(i-1)*Width+j  ] == BLACK) ||
                (pixels_in[(i-1)*Width+j+1] == BLACK) ||
                (pixels_in[ i   *Width+j-1] == BLACK) ||
                (pixels_in[ i   *Width+j  ] == BLACK) ||
                (pixels_in[ i   *Width+j+1] == BLACK) ||
                (pixels_in[(i+1)*Width+j-1] == BLACK) ||
                (pixels_in[(i+1)*Width+j  ] == BLACK) ||
                (pixels_in[(i+1)*Width+j+1] == BLACK) )
                pixels_out[ i   *Width+j  ] = BLACK;
            else
                pixels_out[ i   *Width+j  ] = WHITE;
#else
    uint32_t i, j;
    uint8x16_t v_white = vdupq_n_u8(WHITE);

    for (j = 0; j < Width; j += 16)
    {
        vst1q_u8(pixels_out + j, v_white);
        vst1q_u8(pixels_out + (Height - 1) * Width + j, v_white);
    }

    // hoisting : borne et offset de bord calcules une seule fois
    uint32_t jlim    = Width - 16;
    uint32_t right16 = Width - 16;

    #pragma omp parallel for schedule(static) private(j)
    for (i = 1; i < (Height - 1); i++)
    {
        uint8_t *row_prev = pixels_in  + (i - 1) * Width;
        uint8_t *row_curr = pixels_in  +  i      * Width;
        uint8_t *row_next = pixels_in  + (i + 1) * Width;
        uint8_t *row_out  = pixels_out +  i      * Width;

        vst1q_u8(row_out,             v_white);
        vst1q_u8(row_out + right16,   v_white);

        for (j = 16; j < jlim; j += 16)
        {
            uint8x16_t v_p0 = vld1q_u8(row_prev + j - 1);
            uint8x16_t v_p1 = vld1q_u8(row_prev + j);
            uint8x16_t v_p2 = vld1q_u8(row_prev + j + 1);

            uint8x16_t v_c0 = vld1q_u8(row_curr + j - 1);
            uint8x16_t v_c1 = vld1q_u8(row_curr + j);
            uint8x16_t v_c2 = vld1q_u8(row_curr + j + 1);

            uint8x16_t v_n0 = vld1q_u8(row_next + j - 1);
            uint8x16_t v_n1 = vld1q_u8(row_next + j);
            uint8x16_t v_n2 = vld1q_u8(row_next + j + 1);

            // arbre AND equilibre : profondeur 4 au lieu de 8
            // niveau 1 : 4 AND independants
            uint8x16_t t0 = vandq_u8(v_p0, v_p1);
            uint8x16_t t1 = vandq_u8(v_p2, v_c0);
            uint8x16_t t2 = vandq_u8(v_c1, v_c2);
            uint8x16_t t3 = vandq_u8(v_n0, v_n1);
            // niveau 2
            uint8x16_t t4 = vandq_u8(t0, t1);
            uint8x16_t t5 = vandq_u8(t2, t3);
            // niveaux 3 et 4
            uint8x16_t v_res = vandq_u8(vandq_u8(t4, t5), v_n2);

            vst1q_u8(row_out + j, v_res);
        }
    }
#endif
}

int main()
{
    long duree_totale, duree;
    uint32_t i, nb_pixels, offset_pixels, headers_size;
    type_bitmap bmp_src, bmp_dst;
    uint8_t  *Moy;
    uint16_t *Var;
    char path_src[100], path_dst[100];

    sprintf(path_src, "%s%s%d.bmp", PATH_SRC, IMAGE_NAME, 0);           bmp_open(path_src, &bmp_src);
    headers_size = sizeof(bmp_src.file_header) + sizeof(bmp_src.picture_header);
    nb_pixels = bmp_src.picture_header.image_size;
    offset_pixels = bmp_src.file_header.offset-headers_size;
    memcpy(&(bmp_dst.file_header), &(bmp_src.file_header), headers_size);
    bmp_dst.pixels = (uint8_t*) aligned_alloc(ALIGNMENT, bmp_src.file_header.file_size- headers_size);
    memcpy(bmp_dst.pixels, bmp_src.pixels, offset_pixels);
    Moy = (uint8_t*) aligned_alloc(ALIGNMENT, nb_pixels);
    Var = (uint16_t*) aligned_alloc(ALIGNMENT, nb_pixels << 1);

    SD_Initialization(nb_pixels, bmp_src.pixels+offset_pixels, Moy, Var, bmp_dst.pixels+offset_pixels);
    sprintf(path_dst, "%s%s%d.bmp", PATH_SD, IMAGE_SD, 0);  bmp_write(path_dst, bmp_dst);

    duree_totale = 0;
    for(i=1; i < NB_IMAGES; i++)
    {
        sprintf(path_src, "%s%s%d.bmp", PATH_SRC, IMAGE_NAME, i);           bmp_open(path_src, &bmp_src);
        duree = system_nanoTime();
        Sigma_Delta(nb_pixels, bmp_src.pixels+offset_pixels, Moy, Var, bmp_dst.pixels+offset_pixels);
        duree_totale += system_nanoTime() - duree;
        sprintf(path_dst, "%s%s%d.bmp", PATH_SD, IMAGE_SD, i);  bmp_write(path_dst, bmp_dst);
    }
    printf("Sigma/Delta - Temps de traitement moyen par image : %.2f ns\n", ((double)duree_totale)/(NB_IMAGES - 1));

    Morpho_Initialization(bmp_dst.picture_header.width, bmp_dst.picture_header.height, bmp_dst.pixels+offset_pixels);
    duree_totale = 0;
    for(i=0; i < NB_IMAGES; i++)
    {
        sprintf(path_src, "%s%s%d.bmp", PATH_SD, IMAGE_SD, i);           bmp_open(path_src, &bmp_src);
        duree = system_nanoTime();
        Erosion(bmp_src.picture_header.width, bmp_src.picture_header.height, bmp_src.pixels+offset_pixels, bmp_dst.pixels+offset_pixels);
        duree_totale += system_nanoTime() - duree;
        sprintf(path_dst, "%s%s%d.bmp", PATH_SD, IMAGE_SD_E, i);  bmp_write(path_dst, bmp_dst);
    }
    printf("Erosion     - Temps de traitement moyen par image : %.2f ns\n", ((double)duree_totale)/NB_IMAGES);

    duree_totale = 0;
    for(i=0; i < NB_IMAGES; i++)
    {
        sprintf(path_src, "%s%s%d.bmp", PATH_SD, IMAGE_SD_E, i);           bmp_open(path_src, &bmp_src);
        duree = system_nanoTime();
        Dilatation(bmp_src.picture_header.width, bmp_src.picture_header.height, bmp_src.pixels+offset_pixels, bmp_dst.pixels+offset_pixels);
        duree_totale += system_nanoTime() - duree;
        sprintf(path_dst, "%s%s%d.bmp", PATH_SD, IMAGE_SD_E_D, i);  bmp_write(path_dst, bmp_dst);
    }
    printf("Dilatation  - Temps de traitement moyen par image : %.2f ns\n", ((double)duree_totale)/NB_IMAGES);
    return 0;
}
