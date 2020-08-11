#pragma once
#include "types.hpp"

namespace OneACPose {

  void OneACD(
    const Vec3& dst, const Vec3& src, 
    const Mat32& dst_diff, const Mat32& src_diff, 
    double& scale, Mat3& R, Vec3& t);
    
  void OneACD_Umeyama(
    const Vec3& dst, const Vec3& src, 
    const Mat32& dst_diff, const Mat32& src_diff, 
    double& scale, Mat3& R, Vec3& t);

}