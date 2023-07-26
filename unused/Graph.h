//
// Created by Ashray Gupta on 7/22/23.
//

#ifndef FACTORGRAPHS_GRAPH_H
#define FACTORGRAPHS_GRAPH_H

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

class Graph {
public:
    Eigen::Matrix3d Skew(Eigen::Vector3d w);
    Eigen::SparseMatrix<double> Jacobian(Eigen::VectorXd x, std::function<Eigen::VectorXd(Eigen::VectorXd)> function);

    void run();

    Eigen::VectorXd Step(Eigen::SparseMatrix<double> A, Eigen::VectorXd b);

    Eigen::SparseMatrix<double> WhitenJacobian(Eigen::SparseMatrix<double> J);
};



#endif //FACTORGRAPHS_GRAPH_H
