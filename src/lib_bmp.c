#include "lib_bmp.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

uint8_t bmp_open(char* path, type_bitmap *ptr_bmp)
{
    size_t nb_bytes;
    int file = open(path, O_RDONLY);
    if( file < 0 )
        return -1;

    read(file, &(ptr_bmp->file_header), sizeof(type_bmp_file_header));
    read(file, &(ptr_bmp->picture_header), sizeof(type_bmp_picture_header));

    nb_bytes = (ptr_bmp->picture_header).width*(ptr_bmp->picture_header).height*3;
    ptr_bmp->pixels = (uint8_t*) malloc(nb_bytes);
    read(file, ptr_bmp->pixels, nb_bytes);
    
    close(file);
    return 0;
}

uint8_t bmp_write(char* path, type_bitmap bmp)
{
    size_t nb_bytes;
    int file = open(path, O_CREAT | O_WRONLY, S_IRWXU | S_IRWXG | S_IRWXO);
    if( file < 0 )
        return -1;

    write(file, &(bmp.file_header), sizeof(type_bmp_file_header));
    write(file, &(bmp.picture_header), sizeof(type_bmp_picture_header));
    nb_bytes = bmp.picture_header.width*bmp.picture_header.height*3;
    write(file, bmp.pixels, nb_bytes);

    close(file);
    return 0;
}
