//
// Created by Ashray Gupta on 7/22/23.
//

#include "Graph.h"

void Graph::run() {

    // Least squares problem where h is a generative sensor
    // measurement model and z is the real observation
    //
    //      min Σᵢ||hᵢ(Xᵢ) − zᵢ||²                      (1.1)

    while(true) {
        // Linearize each hᵢ(Xᵢ) using the taylor expansion with
        // Hᵢ as the measurement jacobian for hᵢ(.) around Xᵢ⁰
        //
        //      hᵢ(Xᵢ) = hᵢ(X⁰ᵢ + Δᵢ)
        //             ≈ hᵢ(X⁰ᵢ) + HᵢΔᵢ                     (1.2)
        //
        // Substituting 1.2 into 1.1 results in
        //
        //      min Σᵢ||hᵢ(X⁰ᵢ) + HᵢΔᵢ − zᵢ||²
        //      = min Σᵢ||HᵢΔᵢ − {zᵢ - hᵢ(X⁰ᵢ)}||²          (1.3)
        //      = ∆*
        //
        // We need to perform whitening to get rid of covariances
        // where Σ is the covariance matrix
        //
        //      Aᵢ = Σ⁻⁰ᐧ⁵ * Hᵢ                             (1.4)
        //      bᵢ = Σ⁻⁰ᐧ⁵ * {zᵢ - hᵢ(X⁰ᵢ)}                 (1.5)

        ////WhitenJacobian(J);

        // Combine into least-squares problem form 1.7 which we
        // can SolveFactorGraph using iterative gauss-newton
        //
        //      min Σᵢ||AᵢXᵢ − bᵢ||²                        (1.6)
        //      = min ||AX − b||²                           (1.7)

        ////Eigen::SparseMatrix<double> A;
        ////Eigen::VectorXd b;

        // We can SolveFactorGraph the ∆* from 1.7 using Cholesky factorization
        // in the form AᵀAΔ* = Aᵀb
        ////Eigen::VectorXd delta_star = Step(A, b);

        // We cannot directly optimize over the rotation manifold,
        // so we write rotations linearized around R₀ as
        //
        //      R₀ * e^skew(w)                              (2.1)
        //      where w ∈ ℝ³

        ////R *= Skew(w).exp();
        ////T += t;

    }

}


Eigen::VectorXd Graph::Step(Eigen::SparseMatrix<double> A, Eigen::VectorXd b) {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> LDLT{};
    LDLT.compute(A.transpose() * A);
    return LDLT.solve(A.transpose() * b);
}

Eigen::SparseMatrix<double> Graph::WhitenJacobian(Eigen::SparseMatrix<double> J){


    return J;
}

Eigen::SparseMatrix<double> Graph::Jacobian(Eigen::VectorXd x, std::function<Eigen::VectorXd(Eigen::VectorXd)> function) {
    double epsilon = 1E-6;
    std::vector<Eigen::Triplet<double>> t;

    for (int c = 0; c < x.rows(); c++) {
        Eigen::VectorXd up = x;
        up(c) += epsilon;
        Eigen::VectorXd down = x;
        down(c) -= epsilon;

        // Finite differencing
        // ∂f/∂x ≈ [f(x + ε) − f(x − ε)] / 2ε
        Eigen::VectorXd deriv = (function(up) - function(down)) / (2 * epsilon);

        for (int r = 0; r < deriv.rows(); r++) {
            if(deriv(r) != 0) t.emplace_back(r, c, deriv(r));
        }
    }

    // Sparse Jacobian matrix
    Eigen::SparseMatrix<double> J(x.rows(), x.rows());
    J.setFromTriplets(t.begin(), t.end());

    return J;
}


Eigen::Matrix3d Graph::Skew(Eigen::Vector3d w) {
    Eigen::Matrix3d W = Eigen::MatrixXd::Zero(3, 3);

    // Symmetric skew matrix
    //
    // [0  -wᶻ  wʸ]
    // [wᶻ  0   wˣ]
    // [wʸ  wˣ  0 ]

    W(0, 1) = -w.z();
    W(0, 2) = w.y();

    W(1, 0) = w.z();
    W(1, 2) = -w.x();

    W(2, 0) = -w.y();
    W(2, 1) = w.x();

    return W;
}
