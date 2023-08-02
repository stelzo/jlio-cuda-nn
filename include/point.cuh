#pragma once

#include <iostream>
#include "common.cuh"
#include "math/math.cuh"

namespace jlio
{
  struct PointXYZINormal
  {
    JLIO_INLINE_DEVICE_HOST PointXYZINormal() {}
    JLIO_INLINE_DEVICE_HOST PointXYZINormal(float _x, float _y, float _z, float _intensity, float _normal_x, float _normal_y, float _normal_z, float _curvature) : x(_x), y(_y), z(_z), intensity(_intensity), normal_x(_normal_x), normal_y(_normal_y), normal_z(_normal_z), curvature(_curvature) {}

    float x;
    float y;
    float z;
    float intensity;

    float normal_x;
    float normal_y;
    float normal_z;
    float curvature;

    JLIO_INLINE_DEVICE_HOST bool operator==(const PointXYZINormal &rhs)
    {
      return (x == rhs.x && y == rhs.y && z == rhs.z && intensity == rhs.intensity && normal_x == rhs.normal_x && normal_y == rhs.normal_y && normal_z == rhs.normal_z);
    }

    JLIO_INLINE_DEVICE_HOST friend std::ostream &operator<<(std::ostream &os, const PointXYZINormal &p);

    JLIO_INLINE_DEVICE_HOST float distance(const PointXYZINormal &rhs) const
    {
      return sqrtf((x - rhs.x) * (x - rhs.x) + (y - rhs.y) * (y - rhs.y) + (z - rhs.z) * (z - rhs.z));
    }
  };

  struct OusterPoint
  {
    JLIO_INLINE_DEVICE_HOST OusterPoint() {}
    JLIO_INLINE_DEVICE_HOST OusterPoint(float _x, float _y, float _z, float _intensity, uint32_t _t, uint16_t _reflectivity, uint8_t _ring, uint16_t _ambient, uint32_t _range) : x(_x), y(_y), z(_z), intensity(_intensity), t(_t), reflectivity(_reflectivity), ring(_ring), ambient(_ambient), range(_range) {}

    float x;
    float y;
    float z;
    float intensity;

    // float intensity; // TODO should be inside the union, but in original code it is not
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t ambient;
    uint32_t range;
  };
}
