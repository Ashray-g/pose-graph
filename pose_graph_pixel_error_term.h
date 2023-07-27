//
// Created by Ashray Gupta on 7/24/23.
//

#ifndef FACTORGRAPHS_POSE_GRAPH_PIXEL_ERROR_TERM_H
#define FACTORGRAPHS_POSE_GRAPH_PIXEL_ERROR_TERM_H

#include <utility>

#include "Camera.h"
#include "types.h"

#include "ceres/autodiff_cost_function.h"

class PoseGraphPixelErrorTerm {
public:
    PoseGraphPixelErrorTerm(Eigen::Vector<double, 2> pixel_observed,
                            Eigen::Matrix<double, 2, 2> sqrt_information,
                            Eigen::Matrix<double, 3, 3> intrinsic)
            : pixel_observed(std::move(pixel_observed)),
              sqrt_information(std::move(sqrt_information)),
              intrinsic(std::move(intrinsic)) {}

    template<typename T>
    bool operator()(
            const T *const camera_pose_p_ptr,
            const T *const camera_pose_q_ptr,
            const T *const translation,
            T *residuals_ptr) const {

        //Convert world frame point into camera frame [Xᶜ Yᶜ Zᶜ]
        Eigen::Vector<T, 3> translation_vector_world_frame(translation);

        Eigen::Vector<T, 3> camera_pose_p_world_frame(camera_pose_p_ptr);
        Eigen::Quaternion<T> camera_pose_q_world_frame(camera_pose_q_ptr);

        Eigen::Vector<T, 3> relative_translation_camera_frame =
                translation_vector_world_frame - camera_pose_p_world_frame;
        Eigen::Vector<T, 3> relative_to_camera_transform_camera_frame =
                camera_pose_q_world_frame.inverse() * relative_translation_camera_frame;

        //Intrinsic matrix to convert [Xᶜ Yᶜ Zᶜ] to pixels [x y]
        T x = (T(intrinsic(0, 0)) * relative_to_camera_transform_camera_frame(0)
                    + intrinsic(0, 1) * relative_to_camera_transform_camera_frame(1)
                    + intrinsic(0, 2) * relative_to_camera_transform_camera_frame(2))
                   / relative_to_camera_transform_camera_frame(2);
        T y = (T(intrinsic(1, 1)) * relative_to_camera_transform_camera_frame(1)
                    + intrinsic(1, 2) * relative_to_camera_transform_camera_frame(2))
                   / relative_to_camera_transform_camera_frame(2);

        Eigen::Vector<T, 2> pix;
        pix << x, y;

        //Just don't do distortion stuff inside the ceres stuff (keep outside)

//        residuals_ptr[0] = pix(0) - pixel_observed(0);
//        residuals_ptr[1] = pix(1) - pixel_observed(1);

        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_ptr);
        residuals.template block<2, 1>(0, 0) =
                pix - pixel_observed.template cast<T>();

        // Assuming this is Mahalanobis norm Σ⁻⁰ᐧ⁵
        // From the pose_graph_3d_error_term example
        residuals.applyOnTheLeft(sqrt_information.template cast<T>());

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector<double, 2> &pixel_observed,
                                       const Eigen::Matrix<double, 2, 2> &sqrt_information,
                                       const Eigen::Matrix<double, 3, 3> &intrinsic) {

        // 2 is dimension of residual [Δx Δy]
        // 3 is dimension of the camera pose [X, Y, Z]
        // 4 is dimension of the camera orientation quaternion [x, y, z, w]
        // 3 is dimension of the target pose [X, Y, Z]
        return new ceres::AutoDiffCostFunction<PoseGraphPixelErrorTerm, 2, 3, 4, 3>(
                new PoseGraphPixelErrorTerm(pixel_observed, sqrt_information, intrinsic));
    }

    Eigen::Vector<double, 2> pixel_observed;
    Eigen::Matrix<double, 2, 2> sqrt_information;

    // [fˣ s  cˣ]
    // [0  fʸ cʸ]
    // [0  0  1 ]
    Eigen::Matrix<double, 3, 3> intrinsic;
};

struct ConstraintVision {
    int id_pose;
    int id_landmark;

    Eigen::Vector<double, 2> pixel_coord;

    Eigen::Matrix<double, 2, 2> information;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using VectorOfPixelConstraints = std::vector<ConstraintVision, Eigen::aligned_allocator<ConstraintVision>>;

using MapOfLandmarks =
        std::map<int,
                Eigen::Vector3d,
                std::less<>,
                Eigen::aligned_allocator<std::pair<const int, Eigen::Vector3d>>>;

#endif //FACTORGRAPHS_POSE_GRAPH_PIXEL_ERROR_TERM_H