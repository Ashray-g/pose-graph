#include <iostream>

#include <vector>
#include "unused/Graph.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "pose_graph_3d_error_term.h"
#include "pose_graph_pixel_error_term.h"

#include <random>

#include <fstream>

using namespace ceres::examples;

Eigen::Vector<double, 5> distortionCoeffs;
Eigen::Matrix<double, 3, 3> intrinsic;

bool SolveOptimizationProblem(ceres::Problem *problem) {
    CHECK(problem != nullptr);
    ceres::Solver::Options options;

    options.num_threads = 1;
    options.trust_region_strategy_type = ceres::DOGLEG;
//    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

    options.max_num_iterations = 500;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
//    options.linear_solver_type = ceres::SPARSE_SCHUR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);

    std::cout << summary.FullReport() << '\n';

    return summary.IsSolutionUsable();
}

void SolveFactorGraph(VectorOf3dConstraints &constraints_3d, VectorOfPixelConstraints &constraints_pixel,
                      MapOfPoses *pose_map, MapOfLandmarks *landmark_map) {

    ceres::LossFunction *loss_odom = new ceres::HuberLoss(2.0);
    ceres::LossFunction *loss_vision = new ceres::HuberLoss(2.0);

    ceres::Manifold *manifold = new ceres::EigenQuaternionManifold();

    ceres::Problem problem;
    for (Constraint3d &i: constraints_3d) {
        ceres::CostFunction *cost_function = PoseGraph3dErrorTerm::Create(i.t_be, i.information.llt().matrixL());

        auto a = pose_map->find(i.id_begin);
        auto b = pose_map->find(i.id_end);

        problem.AddResidualBlock(cost_function,
                                 loss_odom,
                                 a->second.p.data(),
                                 a->second.q.coeffs().data(),
                                 b->second.p.data(),
                                 b->second.q.coeffs().data());
        problem.AddParameterBlock(a->second.q.coeffs().data(), 4);
        problem.AddParameterBlock(a->second.p.data(), 3);
        problem.AddParameterBlock(b->second.q.coeffs().data(), 4);
        problem.AddParameterBlock(b->second.p.data(), 3);

        problem.SetManifold(a->second.q.coeffs().data(), manifold);
        problem.SetManifold(b->second.q.coeffs().data(), manifold);
    }

    for (ConstraintVision &i: constraints_pixel) {
        ceres::CostFunction *cost_function = PoseGraphPixelErrorTerm::Create(i.pixel_coord,
                                                                             i.information.llt().matrixL(),
                                                                             intrinsic);

        auto pose = pose_map->find(i.id_pose);
        auto landmark = landmark_map->find(i.id_landmark);

        problem.AddResidualBlock(cost_function,
                                 loss_vision,
                                 pose->second.p.data(),
                                 pose->second.q.coeffs().data(),
                                 landmark->second.data());

        problem.SetParameterBlockConstant(landmark->second.data());

        // Manifolds already set in previous loop for pose quaternions
    }

    SolveOptimizationProblem(&problem);
}

void SetupOdometryInformation(Eigen::Matrix<double, 6, 6> &a) {
    Eigen::Matrix<double, 6, 6> covariance = Eigen::MatrixXd::Identity(6, 6) * 0.02 * 0.02;
    a = covariance.inverse();
}

void SetupVisionInformation(Eigen::Matrix<double, 2, 2> &a) {
    Eigen::Matrix<double, 2, 2> covariance = Eigen::MatrixXd::Identity(2, 2) * 4;
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

    // ground truth for poses: 0, 0.1, 0.2, ..., 4.9
    int n_poses = 50;
    int n_landmarks = 10;

    std::default_random_engine generator_odom(6532); //65 causes issues
    std::default_random_engine generator_pixel{};
    std::normal_distribution<double> distribution_odometry(0.1, 0.02);
    std::normal_distribution<double> distribution_odometry2(0.0, 0.002);
    std::normal_distribution<double> distribution_pixel(0, 2);

    //Add constraints_3d for odometry
    double summation_pose = 0;
    double summation_pose2 = 0;
    for (int i = 0; i < n_poses - 1; i++) {
        Constraint3d constraint;
        constraint.id_begin = i;
        constraint.id_end = i + 1;
        double delta_measurement = distribution_odometry(generator_odom);
        double delta_measurement2 = distribution_odometry2(generator_odom);

        Pose3d constraint_delta;
        constraint_delta.p << delta_measurement, delta_measurement2, 0; // delta measurements fall normally around 0.1 (ground truth)

        constraint.t_be = constraint_delta;
        constraint.information = odometry_factor_information;
        constraints_3d.emplace_back(constraint);

        Pose3d initial_guess_pose;
        initial_guess_pose.p << summation_pose, summation_pose2, 0;
        pose_map->insert(std::pair<int, Pose3d>(i, initial_guess_pose)); // TODO: Verify this is for initial guess?

        summation_pose += delta_measurement;
        summation_pose2 += delta_measurement2;

        if (i == n_poses - 2) {
            Pose3d p;
            p.p << summation_pose, 0, 0;
            pose_map->insert(std::pair<int, Pose3d>(i + 1, p));
        }
    }

    std::ofstream myfile0("landmark_position.txt");

    for (int i = 0; i < n_landmarks; i++) {

        int id = i;
        double mark = (double(i) / n_landmarks) * (5.0);
        Eigen::Vector3d landmark_translation(mark, 0.1, 1); // Ground truth
        landmark_map->insert(std::pair<int, Eigen::Vector3d>(id, landmark_translation));

        myfile0 << landmark_map->find(id)->second(0) << " " << landmark_map->find(id)->second(1) << " " << landmark_map->find(id)->second(2) << "\n";

        for (int j = 0; j < n_poses; j++) {
            Eigen::Vector3d v(j * 0.1, 0, 0); // Ground truth camera pose

            auto frame_coord = frame_coords(landmark_translation, v,
                                            pose_map->find(0)->second.q); // Measurement at ground truth

            //Check if landmark is in frame (for simulation)
            if (frame_coord(0) >= 0 && frame_coord(0) <= 1280 && frame_coord(1) >= 0 && frame_coord(1) <= 960) {

                frame_coord(0) += distribution_pixel(generator_pixel); // Add noise
                frame_coord(1) += distribution_pixel(generator_pixel); // Add noise

                ConstraintVision vision_constraint;
                vision_constraint.id_landmark = id;
                vision_constraint.id_pose = j;

                vision_constraint.information = vision_factor_information;
                vision_constraint.pixel_coord = frame_coord;

                constraints_pixel.emplace_back(vision_constraint);
            }
        }
    }
    myfile0.close();

    std::ofstream myfile1("initial_guess.txt");
    for (int i = 0; i < n_poses; i++) {
        myfile1 << pose_map->find(i)->second.p(0) << " " << pose_map->find(i)->second.p(1) << " "
                << pose_map->find(i)->second.p(2) << "\n";
    }
    myfile1.close();

    SolveFactorGraph(constraints_3d, constraints_pixel, pose_map, landmark_map);

    std::ofstream myfile2("solved_out.txt");
    for (int i = 0; i < n_poses; i++) {
        myfile2 << pose_map->find(i)->second.p(0) << " " << pose_map->find(i)->second.p(1) << " "
                << pose_map->find(i)->second.p(2) << "\n";
    }
    myfile2.close();
}


int main() {
    test();

    return 0;
}

