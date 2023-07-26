#include <iostream>

#include <vector>
#include "unused/Graph.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

#include "pose_graph_3d_error_term.h"
#include "pose_graph_pixel_error_term.h"

#include <random>

using namespace ceres::examples;

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

        std::cout << a->second.p.data() << "\n\n";

        problem.SetManifold(a->second.q.coeffs().data(), manifold);
        problem.SetManifold(b->second.q.coeffs().data(), manifold);
    }

    //TODO clean these and initialize somewhere else
    Eigen::Vector<double, 5> dist;
    Eigen::Matrix<double, 3, 3> intrin;

    for (ConstraintVision &i: constraints_pixel) {
        ceres::CostFunction *cost_function = PoseGraphPixelErrorTerm::Create(i.pixel_coord,
                                                                             i.information.llt().matrixL(), dist,
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
    Eigen::Matrix<double, 2, 2> covariance = Eigen::MatrixXd::Identity(2, 2) * 1;
    a = covariance.inverse();
}

void test() {
    Eigen::Matrix<double, 6, 6> odometry_factor_information;
    Eigen::Matrix<double, 2, 2> vision_factor_information;

    SetupOdometryInformation(odometry_factor_information);
    SetupVisionInformation(vision_factor_information);

    VectorOf3dConstraints constraints_3d;
    VectorOfPixelConstraints constraints_pixel;
    auto *pose_map = new MapOfPoses();
    auto *landmark_map = new MapOfLandmarks();

    int n = 50;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.1, 0.01);

    double last = 0;

    //Add constraints_3d for odometry
    for (int i = 0; i < n; i++) {
        Constraint3d constraint;
        constraint.id_begin = i;
        constraint.id_end = i + 1;

        Pose3d p1;
        p1.p(0) = 0.1;
        p1.p(1) = 0;
        p1.p(2) = 0;

        constraint.t_be = p1;
        constraint.information = odometry_factor_information;

        constraints_3d.emplace_back(constraint);

        Pose3d p;

        p.p(0) = last + distribution(generator);
        p.p(1) = 0;
        p.p(2) = 0;
        pose_map->insert(std::pair<int, Pose3d>(i, p));
        last = p.p(0);
    }

    //TODO: Add constaints_vision for landmarks next

    Pose3d p;
    p.p(0) = n * 0.11;
    p.p(1) = 0;
    p.p(2) = 0;
    pose_map->insert(std::pair<int, Pose3d>(n, p));

    for (int i = 0; i <= n; i++) {
        std::cout << pose_map->find(i)->second.p(0) << " ";
    }
    std::cout << "\n\n";

    SolveFactorGraph(constraints_3d, constraints_pixel, pose_map, landmark_map);

    std::cout << "\n\n";
    for (int i = 0; i <= n; i++) {
        std::cout << pose_map->find(i)->second.p(0) << " ";
    }
}


int main() {
    test();

    return 0;
}

