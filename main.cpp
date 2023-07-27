#include <iostream>

#include <vector>
#include "unused/Graph.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "pose_graph_3d_error_term.h"
#include "pose_graph_pixel_error_term.h"

#include <random>

using namespace ceres::examples;

Eigen::Vector<double, 5> distortionCoeffs;
Eigen::Matrix<double, 3, 3> intrinsic;

bool SolveOptimizationProblem(ceres::Problem *problem) {
    CHECK(problem != nullptr);
    ceres::Solver::Options options;

    options.num_threads = 1;
    options.trust_region_strategy_type = ceres::DOGLEG;

    options.max_num_iterations = 50;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);

    std::cout << summary.FullReport() << '\n';

    return summary.IsSolutionUsable();
}

void SolveFactorGraph(VectorOf3dConstraints &constraints_3d, VectorOfPixelConstraints &constraints_pixel,
                      MapOfPoses *pose_map, MapOfLandmarks *landmark_map) {
    ceres::LossFunction *loss = new ceres::HuberLoss(2.0);
    ceres::Manifold *manifold = new ceres::EigenQuaternionManifold();

    ceres::Problem problem;

    for (Constraint3d &i: constraints_3d) {
        ceres::CostFunction *cost_function = PoseGraph3dErrorTerm::Create(i.t_be, i.information.llt().matrixL());

        auto a = pose_map->find(i.id_begin);
        auto b = pose_map->find(i.id_end);

        problem.AddResidualBlock(cost_function,
                                 loss,
                                 a->second.p.data(),
                                 a->second.q.coeffs().data(),
                                 b->second.p.data(),
                                 b->second.q.coeffs().data());

        problem.SetManifold(a->second.q.coeffs().data(), manifold);
        problem.SetManifold(b->second.q.coeffs().data(), manifold);
    }

    //TODO clean these and initialize somewhere else
    Eigen::Vector<double, 5> dist;
    Eigen::Matrix<double, 3, 3> intrin;

    if (true) {
        for (ConstraintVision &i: constraints_pixel) {
            ceres::CostFunction *cost_function = PoseGraphPixelErrorTerm::Create(i.pixel_coord,
                                                                                 i.information.llt().matrixL(),
                                                                                 intrin);

            auto pose = pose_map->find(i.id_pose);
            auto landmark = landmark_map->find(i.id_landmark);

            problem.AddResidualBlock(cost_function,
                                     loss,
                                     pose->second.p.data(),
                                     pose->second.q.coeffs().data(),
                                     landmark->second.data());

            // Manifolds already set in previous loop for pose quaternions
        }
    }

    //TODO: set landmarks constant?

    problem.SetParameterBlockConstant(pose_map->find(constraints_3d[0].id_begin)->second.p.data());
    problem.SetParameterBlockConstant(pose_map->find(constraints_3d[0].id_begin)->second.q.coeffs().data());

    SolveOptimizationProblem(&problem);
}

void SetupOdometryInformation(Eigen::Matrix<double, 6, 6> &a) {
    Eigen::Matrix<double, 6, 6> covariance = Eigen::MatrixXd::Identity(6, 6) * 1e-1;
    a = covariance.inverse();
}

void SetupVisionInformation(Eigen::Matrix<double, 2, 2> &a) {
    Eigen::Matrix<double, 2, 2> covariance = Eigen::MatrixXd::Identity(2, 2) * 1e-1;
    a = covariance.inverse();

    distortionCoeffs << 0.06339634695488064,
            -0.06963304849427762,
            0.0008610391358151331,
            -0.0013580659961053847,
            -0.010787937240417867;

    intrinsic << 714.4774572094963, 0, 619.9129445218975,
            0, 1172.20100070466, 446.883179888407,
            0, 0, 1;
}

Eigen::Vector2d frame_coords(const Eigen::Vector3d &translation_vector_world_frame,
                             const Eigen::Vector3d &camera_pose_p_world_frame,
                             const Eigen::Quaterniond &camera_pose_q_world_frame) {

    Eigen::Vector3d relative_translation_camera_frame =
            translation_vector_world_frame - camera_pose_p_world_frame;
    Eigen::Vector3d relative_to_camera_transform_camera_frame =
            camera_pose_q_world_frame.inverse() * relative_translation_camera_frame;

    //Intrinsic matrix to convert [Xᶜ Yᶜ Zᶜ] to pixels [x y]
    double x = (intrinsic(0, 0) * relative_to_camera_transform_camera_frame(0)
                + intrinsic(0, 1) * relative_to_camera_transform_camera_frame(1)
                + intrinsic(0, 2) * relative_to_camera_transform_camera_frame(2))
               / relative_to_camera_transform_camera_frame(2);
    double y = (intrinsic(1, 1) * relative_to_camera_transform_camera_frame(1)
                + intrinsic(1, 2) * relative_to_camera_transform_camera_frame(2))
               / relative_to_camera_transform_camera_frame(2);

    Eigen::Vector<double, 2> pix;
    pix << x, y;

    return pix;
}

// I NEED to rewrite this in the future, just for testing right now
void test() {
    Eigen::Matrix<double, 6, 6> odometry_factor_information;
    Eigen::Matrix<double, 2, 2> vision_factor_information;

    SetupOdometryInformation(odometry_factor_information);
    SetupVisionInformation(vision_factor_information);

    VectorOf3dConstraints constraints_3d;
    VectorOfPixelConstraints constraints_pixel;

    auto *pose_map = new MapOfPoses();
    auto *landmark_map = new MapOfLandmarks();

    int n_poses = 50;
    int n_landmarks = 10;

    std::default_random_engine generator_odom(1);
    std::default_random_engine generator_pixel(1);
    std::normal_distribution<double> distribution_odometry(0.1, 0.005);
    std::normal_distribution<double> distribution_pixel(0, 100);

    //Add constraints_3d for odometry
    double last = 0;
    for (int i = 0; i < n_poses; i++) {
        Constraint3d constraint;
        constraint.id_begin = i;
        constraint.id_end = i + 1;

        Pose3d p1;
        p1.p << 0.1, 0, 0;

        constraint.t_be = p1;
        constraint.information = odometry_factor_information;
        constraints_3d.emplace_back(constraint);

        Pose3d p;
        last += distribution_odometry(generator_odom);
        p.p << last, 0, 0;
        pose_map->insert(std::pair<int, Pose3d>(i, p));
    }
    Pose3d p;
    p.p << last + distribution_odometry(generator_odom), 0, 0;
    pose_map->insert(std::pair<int, Pose3d>(n_poses, p));

    for (int i = 0; i < n_landmarks; i++) {

        int id = i;
        //Should run them alongside the poses?
        Eigen::Vector3d landmark_translation((double(i) / n_landmarks) * (n_poses / 10.0), 0, 1);
        landmark_map->insert(std::pair<int, Eigen::Vector3d>(id, landmark_translation));

        for (int j = 0; j < n_poses; j++) {
            auto frame_coord = frame_coords(landmark_translation, pose_map->find(0)->second.p,
                                            pose_map->find(0)->second.q);


            if (frame_coord(0) >= 0 && frame_coord(0) <= 1280 && frame_coord(1) >= 0 && frame_coord(1) <= 960) {

                std::cout << frame_coord(0);
                frame_coord(0) += distribution_pixel(generator_pixel);
                std::cout << "-> " << frame_coord(0) << "\n\n";
                frame_coord(1) += distribution_pixel(generator_pixel);

                ConstraintVision vision_constraint;
                vision_constraint.id_landmark = id;
                vision_constraint.id_pose = j;

                vision_constraint.information = vision_factor_information;
                vision_constraint.pixel_coord = frame_coord;

                constraints_pixel.emplace_back(vision_constraint);
            }
        }
    }


    for (int i = 0; i <= n_poses; i++) {
        std::cout << pose_map->find(i)->second.p(0) << " ";
    }
    std::cout << "\n\n";

    SolveFactorGraph(constraints_3d, constraints_pixel, pose_map, landmark_map);

    std::cout << "\n\n";
    for (int i = 0; i <= n_poses; i++) {
        std::cout << pose_map->find(i)->second.p(0) << " ";
    }
}


int main() {
    test();

    return 0;
}

