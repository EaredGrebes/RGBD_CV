#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <math_constants.h>
#include <opencv2/cudaarithm.hpp>


//-----------------------------------------------------------------------------
__device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b) {

    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}


//-----------------------------------------------------------------------------
__device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b) {

    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}


//-----------------------------------------------------------------------------
__device__ __forceinline__ float3 operator/(const float3 &a, const float &b) {

    return make_float3(a.x/b, a.y/b, a.z/b);
}


//-----------------------------------------------------------------------------
__device__ __forceinline__ float3 operator*(const float3 &a, const float &b) {

    return make_float3(a.x*b, a.y*b, a.z*b);
}


//-----------------------------------------------------------------------------
__device__ __forceinline__ float sign(const float &x) { 

    float t = x<0 ? -1 : 0;
    return x > 0 ? 1 : t;
}


//-----------------------------------------------------------------------------
__device__ __forceinline__ float3 normalize(const float3 &a) { 

    float norm = sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
    return (a / norm);
}


//-----------------------------------------------------------------------------
__device__ __forceinline__ float dot(const float3 &a, const float3 &b) { 

    float tmp = (a.x*b.x + a.y*b.y + a.z*b.z);
    return tmp;
}


//-----------------------------------------------------------------------------
__device__ __forceinline__ float3 rotateTranslate(float *rotation,     // inputs
                                                  float *translation, 
                                                  float3 posInA)
{
    float3 posInB;

    posInB.x = rotation[0] * posInA.x + rotation[3] * posInA.y + rotation[6] * posInA.z + translation[0];
    posInB.y = rotation[1] * posInA.x + rotation[4] * posInA.y + rotation[7] * posInA.z + translation[1];
    posInB.z = rotation[2] * posInA.x + rotation[5] * posInA.y + rotation[8] * posInA.z + translation[2];

    return posInB;
}

#endif

