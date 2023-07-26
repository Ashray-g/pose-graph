//
// Created by Ashray Gupta on 7/25/23.
//

#ifndef FACTORGRAPHS_CAMERA_H
#define FACTORGRAPHS_CAMERA_H

#include <Eigen/Core>
#include <utility>

template <typename T>
class Camera {
public:
    // [k₁  k₂  k₃  p₁  p₂]
    Eigen::Vector<T, 2> distortionCoeffs;

    // [fˣ s  cˣ]
    // [0  fʸ cʸ]
    // [0  0  1 ]
    Eigen::Matrix<T, 3, 3> intrinsic;

    // Width x Height (pixels)
    Eigen::Vector2d frame_size;

    Camera(Eigen::Vector<T, 2>&& distortionCoeffs, Eigen::Matrix<T, 3, 3>&& intrinsic, Eigen::Vector2d&& frame_size)
            : distortionCoeffs(std::move(distortionCoeffs)),
              frame_size(std::move(frame_size)),
              intrinsic(std::move(intrinsic)) {}

    Camera()= default;

    // [x]   [fˣ s  cˣ]   [X]
    // [y] = [0  fʸ cʸ] * [Y]
    // [w]   [0  0  1 ]   [Z]
    void world_to_frame(Eigen::Vector<T, 3>& xyz, Eigen::Vector<T, 2>& pix){
        T x = intrinsic(0, 0) * xyz(0) + intrinsic(0, 1) * xyz(1) + intrinsic(0, 2) * xyz(2);
        T y = intrinsic(1, 0) * xyz(0) + intrinsic(1, 1) * xyz(1) + intrinsic(1, 2) * xyz(2);
        T w = xyz(2);

        pix << x, y;
    }

};


#endif //FACTORGRAPHS_CAMERA_H
