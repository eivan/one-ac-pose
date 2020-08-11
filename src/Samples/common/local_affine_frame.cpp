#include "local_affine_frame.hpp"

namespace common {

  void Feature_LAF_D::as_3D(
    const CameraPtr& cam, 
    OneACPose::Vec3& Y,
    OneACPose::Mat32& dY_dx) const
  {
    // 3D LAF 0 (q0, Q0)
    const auto [q0, dq0_dx0] = cam->q_gradient(x);
    // 3D-projected affine shape, aka q0_dx
    const OneACPose::Mat32 Q0 = dq0_dx0 * M;
    // combine 3D LAF 0 with depth and its derivatives
    Y.noalias() = q0 * lambda;
    dY_dx.noalias() = q0 * dlambda_dx + Q0 * lambda;
  }

}