#ifndef UWB_LOCALIZATION_ALGEBRAIC_INITIALIZATION_H_
#define UWB_LOCALIZATION_ALGEBRAIC_INITIALIZATION_H_

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <stdexcept>
#include <vector>
#include "sophus/se3.hpp"


namespace uwb_localization
{


struct Measurement {
    rclcpp::Time timestamp;   // Time of the measurement
    std::string tag_id;       // ID of the tag
    std::string anchor_id;    // ID of the anchor
    double distance;          // Measured distance (in meters)

    // Position of the tag in the UAV odometry frame.
    Eigen::Vector3d tag_odom_pose;
    // Position of the anchor in the AGV odometry frame.
    Eigen::Vector3d anchor_odom_pose;

    //Store the cumulative displacement (e.g., total distance traveled) for each sensor at the measurement time.
    double tag_cumulative_distance;
    double anchor_cumulative_distance;
};



// This function builds the full lifted matrix M_full = [ M | A | -ε ]
// where:
// - M (size: (m+1)×10) is the block from the quadratic monomials in the quaternion,
// - A (size: (m+1)×6) multiplies [p0; r0],
// - and -ε (size: (m+1)×1) is built from the auxiliary terms.
// Here m is the number of distance measurements in your minimal subset.
inline Eigen::MatrixXd buildLiftedMatrix(const std::vector<Measurement> &subset) {

    // Let m be the number of measurements; the full system has (m+1) rows.
    size_t m = subset.size();
    size_t rows = m + 1;
    // The unknown vector is of size 17: 10 for the quaternion monomials, 6 for [p0; r0], and 1 for ρ.
    size_t cols = 17;
    Eigen::MatrixXd M_full = Eigen::MatrixXd::Zero(rows, cols);

    // Use the distance from the first measurement as the reference (d0).
    double d0 = subset[0].distance;

    // Loop over each measurement to fill rows 0 to m-1.
    for (size_t k = 0; k < m; ++k) {
        const Measurement &meas = subset[k];
        double d_k = meas.distance;

        Eigen::Vector3d p1 = meas.tag_odom_pose;    // tag position in UAV odom frame
        Eigen::Vector3d p2 = meas.anchor_odom_pose;   // anchor position in AGV odom frame

        // For convenience, let p1 = [u, v, w]ᵀ and p2 = [x, y, z]ᵀ.
        double u = p1(0), v = p1(1), w = p1(2);
        double x = p2(0), y = p2(1), z = p2(2);

        // --- Build the quaternion block (columns 0..9) ---
        // Using the standard expansion of p1ᵀ R(q) p2, we set:
        Eigen::VectorXd row_quat(10);
        row_quat(0) = u*x - v*y - w*z;             // coefficient for q1²
        row_quat(1) = u*y + v*x;                     // coefficient for q1q2
        row_quat(2) = u*z + w*x;                     // coefficient for q1q3
        row_quat(3) = -v*z + w*y;                    // coefficient for q1q4
        row_quat(4) = -u*x + v*y - w*z;              // coefficient for q2²
        row_quat(5) = v*z + w*y;                     // coefficient for q2q3
        row_quat(6) = u*z - w*x;                     // coefficient for q2q4
        row_quat(7) = -u*x - v*y + w*z;              // coefficient for q3²
        row_quat(8) = -u*y + v*x;                    // coefficient for q3q4
        row_quat(9) = u*x + v*y + w*z;               // coefficient for q4²

        // Place this 1×10 row in M_full (columns 0 to 9) for row k.
        M_full.row(k).segment(0, 10) = row_quat.transpose();

        // --- Build the A block (columns 10..15) ---
        // For each measurement, A(k,:) = [ p1ᵀ,  –p2ᵀ ].
        Eigen::RowVectorXd row_A(6);
        row_A << p1.transpose(), (-p2).transpose();
        M_full.row(k).segment(10, 6) = row_A;

        // --- Build the -ε block (column 16) ---
        // Compute εₖ = 0.5*(d0² + ∥p1∥² + ∥p2∥² – dₖ²)
        double eps = 0.5 * (d0*d0 + p1.squaredNorm() + p2.squaredNorm() - d_k*d_k);
        M_full(k, 16) = -eps;
    }

    // --- Last row: the unit–quaternion constraint ---
    // According to the paper, set the quaternion block as [1 0 0 0 1 0 0 1 0 1],
    // and zeros in the A block; for –ε, set –1.
    Eigen::RowVectorXd last_quat(10);
    last_quat << 1, 0, 0, 0, 1, 0, 0, 1, 0, 1;
    M_full.row(m).segment(0, 10) = last_quat;
    M_full.row(m).segment(10, 6).setZero();
    M_full(m, 16) = -1;

    return M_full;
}


// Example: robust quaternion recovery
inline Eigen::Vector4d recoverQuaternionFromMonomials(const Eigen::VectorXd &qMonomials) {
  if (qMonomials.size() != 10) {
    throw std::runtime_error("qMonomials must be of size 10");
  }
  Eigen::Matrix4d Q;
  Q(0,0) = qMonomials(0);
  Q(0,1) = qMonomials(1);
  Q(0,2) = qMonomials(2);
  Q(0,3) = qMonomials(3);
  Q(1,0) = qMonomials(1);
  Q(1,1) = qMonomials(4);
  Q(1,2) = qMonomials(5);
  Q(1,3) = qMonomials(6);
  Q(2,0) = qMonomials(2);
  Q(2,1) = qMonomials(5);
  Q(2,2) = qMonomials(7);
  Q(2,3) = qMonomials(8);
  Q(3,0) = qMonomials(3);
  Q(3,1) = qMonomials(6);
  Q(3,2) = qMonomials(8);
  Q(3,3) = qMonomials(9);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(Q);
  Eigen::Vector4d q = es.eigenvectors().col(3);
  q.normalize();
  if(q(3) < 0)
      q = -q;
  return q;
}


// Helper: Given the nullspace of dimension N, map a pair (i,j) with i<=j
// to an index in the vector of length N(N+1)/2.
inline int pairIndex(int i, int j, int N) {
    return i * N - (i * (i - 1)) / 2 + (j - i);
}

// Helper: Add a bilinear term coeff * (x[idx1]*x[idx2]) to the given row in L.
// V is the (17 x N) nullspace basis of the lifted system.
inline void addBilinearTerm(Eigen::MatrixXd &L, int row, double coeff, int idx1, int idx2, const Eigen::MatrixXd &V) {
    int N = V.cols();
    for (int i = 0; i < N; i++) {
        for (int j = i; j < N; j++) {
            double term = 0.0;
            if (i == j) {
                term = coeff * V(idx1, i) * V(idx2, i);
            } else {
                term = coeff * (V(idx1, i) * V(idx2, j) + V(idx1, j) * V(idx2, i));
            }
            int pos = pairIndex(i, j, N);
            L(row, pos) += term;
        }
    }
}


// --------------------------------------------------------------------------
// Compute the L matrix (27 x [N(N+1)/2]) according to eq. (43) of Trawny et al.
// V is the (17 x N) nullspace basis of the lifted system M_full.
// The ordering of x is as follows:
//   x[0..9]   : Quaternion monomials: [q₁², q₁q₂, q₁q₃, q₁q₄, q₂², q₂q₃, q₂q₄, q₃², q₃q₄, q₄²]
//   x[10..12] : p₀ ∈ ℝ³
//   x[13..15] : r₀ ∈ ℝ³  (with the intended relation r₀ = C(q)ᵀ p₀)
//   x[16]     : ρ
// The 27 constraints consist of:
//   (a) 20 quaternion consistency constraints,
//   (b) 6 constraints from the p₀–r₀ relation, and
//   (c) 1 scaling constraint.
inline Eigen::MatrixXd computeLMatrix(const Eigen::MatrixXd &V) {
    
    int N = V.cols();
    int numLambda = N * (N + 1) / 2;
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(27, numLambda);
    int row = 0;
    
    // ---------- (a) 20 Quaternion Consistency Constraints ----------
    // Here we use the following constraints (using zero-based indexing in the quaternion block):
    // 1.  x[0]*x[4] - (x[1])² = 0
    addBilinearTerm(L, row++, 1.0, 0, 4, V);
    addBilinearTerm(L, row++, -1.0, 1, 1, V);
    // 2.  x[0]*x[7] - (x[2])² = 0
    addBilinearTerm(L, row++, 1.0, 0, 7, V);
    addBilinearTerm(L, row++, -1.0, 2, 2, V);
    // 3.  x[0]*x[9] - (x[3])² = 0
    addBilinearTerm(L, row++, 1.0, 0, 9, V);
    addBilinearTerm(L, row++, -1.0, 3, 3, V);
    // 4.  x[4]*x[7] - (x[5])² = 0
    addBilinearTerm(L, row++, 1.0, 4, 7, V);
    addBilinearTerm(L, row++, -1.0, 5, 5, V);
    // 5.  x[4]*x[9] - (x[6])² = 0
    addBilinearTerm(L, row++, 1.0, 4, 9, V);
    addBilinearTerm(L, row++, -1.0, 6, 6, V);
    // 6.  x[7]*x[9] - (x[8])² = 0
    addBilinearTerm(L, row++, 1.0, 7, 9, V);
    addBilinearTerm(L, row++, -1.0, 8, 8, V);
    // 7.  x[0]*x[5] - x[1]*x[2] = 0
    addBilinearTerm(L, row++, 1.0, 0, 5, V);
    addBilinearTerm(L, row++, -1.0, 1, 2, V);
    // 8.  x[0]*x[6] - x[1]*x[3] = 0
    addBilinearTerm(L, row++, 1.0, 0, 6, V);
    addBilinearTerm(L, row++, -1.0, 1, 3, V);
    // 9.  x[0]*x[8] - x[2]*x[3] = 0
    addBilinearTerm(L, row++, 1.0, 0, 8, V);
    addBilinearTerm(L, row++, -1.0, 2, 3, V);
    // 10. x[1]*x[7] - x[2]*x[5] = 0
    addBilinearTerm(L, row++, 1.0, 1, 7, V);
    addBilinearTerm(L, row++, -1.0, 2, 5, V);
    // 11. x[1]*x[9] - x[3]*x[6] = 0
    addBilinearTerm(L, row++, 1.0, 1, 9, V);
    addBilinearTerm(L, row++, -1.0, 3, 6, V);
    // 12. x[2]*x[9] - x[3]*x[8] = 0
    addBilinearTerm(L, row++, 1.0, 2, 9, V);
    addBilinearTerm(L, row++, -1.0, 3, 8, V);
    // 13. x[1]*x[8] - x[2]*x[7] = 0
    addBilinearTerm(L, row++, 1.0, 1, 8, V);
    addBilinearTerm(L, row++, -1.0, 2, 7, V);
    // 14. x[2]*x[6] - x[1]*x[5] = 0
    addBilinearTerm(L, row++, 1.0, 2, 6, V);
    addBilinearTerm(L, row++, -1.0, 1, 5, V);
    // 15. x[3]*x[7] - x[1]*x[9] = 0
    addBilinearTerm(L, row++, 1.0, 3, 7, V);
    addBilinearTerm(L, row++, -1.0, 1, 9, V);
    // 16. x[3]*x[8] - x[2]*x[7] = 0
    addBilinearTerm(L, row++, 1.0, 3, 8, V);
    addBilinearTerm(L, row++, -1.0, 2, 7, V);
    // 17. x[5]*x[9] - x[6]*x[8] = 0
    addBilinearTerm(L, row++, 1.0, 5, 9, V);
    addBilinearTerm(L, row++, -1.0, 6, 8, V);
    // 18. x[1]*x[8] - x[3]*x[5] = 0
    addBilinearTerm(L, row++, 1.0, 1, 8, V);
    addBilinearTerm(L, row++, -1.0, 3, 5, V);
    // 19. x[2]*x[7] - x[3]*x[4] = 0
    addBilinearTerm(L, row++, 1.0, 2, 7, V);
    addBilinearTerm(L, row++, -1.0, 3, 4, V);
    // 20. x[2]*x[6] - x[1]*x[7] = 0
    addBilinearTerm(L, row++, 1.0, 2, 6, V);
    addBilinearTerm(L, row++, -1.0, 1, 7, V);
    
    // ---------- (b) 6 p₀–r₀ Constraints ----------
    // We now enforce the relation r₀ = Cᵀ p₀.
    // With the rotation matrix expressed in terms of the quaternion–monomials:
    //   R(0,0) = 2*(x[0]+x[4]) - ρ,   R(1,0) = 2*(x[5]+x[3]),   R(2,0) = 2*(x[6]-x[2]),
    //   R(0,1) = 2*(x[5]-x[3]),         R(1,1) = 2*(x[0]+x[7]) - ρ, R(2,1) = 2*(x[8]-x[1]),
    //   R(0,2) = 2*(x[2]+x[6]),         R(1,2) = 2*(x[8]+x[1]),     R(2,2) = 2*(x[0]+x[9]) - ρ.
    //
    // The six constraints are then:
    // For i = 0,1,2:
    //   (i) r₀(i) - [R(0,i)*p₀(0) + R(1,i)*p₀(1) + R(2,i)*p₀(2)] = 0, and
    //   (ii) p₀(i) - [R(i,0)*r₀(0) + R(i,1)*r₀(1) + R(i,2)*r₀(2)] = 0.
    //
    // --- Constraint E0: (i) for i = 0 ---
    // r₀(0) = x[13], p₀(0)=x[10], p₀(1)=x[11], p₀(2)=x[12].
    // R(0,0)=2*(x[0]+x[4]) - x[16], R(1,0)=2*(x[5]+x[3]), R(2,0)=2*(x[6]-x[2]).
    // So: E0 = x[13] - { [2*(x[0]+x[4]) - x[16]]*x[10] + 2*(x[5]+x[3])*x[11] + 2*(x[6]-x[2])*x[12] } = 0.
    // Write this as a sum of bilinear terms:
    addBilinearTerm(L, row, 1.0, 13, 16, V);            // x[13]*x[16]
    addBilinearTerm(L, row, -2.0, 0, 10, V);             // -2*x[0]*x[10]
    addBilinearTerm(L, row, -2.0, 4, 10, V);             // -2*x[4]*x[10]
    addBilinearTerm(L, row, +1.0, 10, 16, V);            // +1*x[10]*x[16]  [for the -(-x[16])*x[10]]
    addBilinearTerm(L, row, -2.0, 5, 11, V);             // -2*x[5]*x[11]
    addBilinearTerm(L, row, -2.0, 3, 11, V);             // -2*x[3]*x[11]
    addBilinearTerm(L, row, -2.0, 6, 12, V);             // -2*x[6]*x[12]
    addBilinearTerm(L, row, +2.0, 2, 12, V);             // +2*x[2]*x[12]
    row++;
    
    // --- Constraint E1: (i) for i = 1 ---
    // r₀(1)=x[14], p₀ as before.
    // R(0,1)=2*(x[5]-x[3]), R(1,1)=2*(x[0]+x[7]) - x[16], R(2,1)=2*(x[8]-x[1]).
    // E1 = x[14] - { 2*(x[5]-x[3])*x[10] + [2*(x[0]+x[7]) - x[16]]*x[11] + 2*(x[8]-x[1])*x[12] } = 0.
    addBilinearTerm(L, row, 1.0, 14, 16, V);            // x[14]*x[16]
    addBilinearTerm(L, row, -2.0, 5, 10, V);             // -2*x[5]*x[10]
    addBilinearTerm(L, row, +2.0, 3, 10, V);             // +2*x[3]*x[10]
    addBilinearTerm(L, row, -2.0, 0, 11, V);             // -2*x[0]*x[11]
    addBilinearTerm(L, row, -2.0, 7, 11, V);             // -2*x[7]*x[11]
    addBilinearTerm(L, row, +1.0, 11, 16, V);            // +1*x[11]*x[16]
    addBilinearTerm(L, row, -2.0, 8, 12, V);             // -2*x[8]*x[12]
    addBilinearTerm(L, row, +2.0, 1, 12, V);             // +2*x[1]*x[12]
    row++;
    
    // --- Constraint E2: (i) for i = 2 ---
    // R(0,2)=2*(x[2]+x[6]), R(1,2)=2*(x[8]+x[1]), R(2,2)=2*(x[0]+x[9]) - x[16].
    // E2 = x[15] - { 2*(x[2]+x[6])*x[10] + 2*(x[8]+x[1])*x[11] + [2*(x[0]+x[9]) - x[16]]*x[12] } = 0.
    addBilinearTerm(L, row, 1.0, 15, 16, V);            // x[15]*x[16]
    addBilinearTerm(L, row, -2.0, 2, 10, V);             // -2*x[2]*x[10]
    addBilinearTerm(L, row, -2.0, 6, 10, V);             // -2*x[6]*x[10]
    addBilinearTerm(L, row, -2.0, 8, 11, V);             // -2*x[8]*x[11]
    addBilinearTerm(L, row, -2.0, 1, 11, V);             // -2*x[1]*x[11]
    addBilinearTerm(L, row, -2.0, 0, 12, V);             // -2*x[0]*x[12]
    addBilinearTerm(L, row, -2.0, 9, 12, V);             // -2*x[9]*x[12]
    addBilinearTerm(L, row, +1.0, 12, 16, V);            // +1*x[12]*x[16]
    row++;
    
    // Now the transposed set of constraints (p₀ = C r₀):
    // --- Constraint E3: for i = 0 ---
    // E3 = x[10] - { [2*(x[0]+x[4]) - x[16]]*x[13] + 2*(x[5]-x[3])*x[14] + 2*(x[2]+x[6])*x[15] } = 0.
    addBilinearTerm(L, row, 1.0, 10, 16, V);            // x[10]*x[16]
    addBilinearTerm(L, row, -2.0, 0, 13, V);             // -2*x[0]*x[13]
    addBilinearTerm(L, row, -2.0, 4, 13, V);             // -2*x[4]*x[13]
    addBilinearTerm(L, row, +1.0, 13, 16, V);            // +1*x[13]*x[16]
    addBilinearTerm(L, row, -2.0, 5, 14, V);             // -2*x[5]*x[14]
    addBilinearTerm(L, row, +2.0, 3, 14, V);             // +2*x[3]*x[14]
    addBilinearTerm(L, row, -2.0, 2, 15, V);             // -2*x[2]*x[15]
    addBilinearTerm(L, row, -2.0, 6, 15, V);             // -2*x[6]*x[15]
    row++;
    
    // --- Constraint E4: for i = 1 ---
    // E4 = x[11] - { 2*(x[5]+x[3])*x[13] + [2*(x[0]+x[7]) - x[16]]*x[14] + 2*(x[8]+x[1])*x[15] } = 0.
    addBilinearTerm(L, row, 1.0, 11, 16, V);            // x[11]*x[16]
    addBilinearTerm(L, row, -2.0, 5, 13, V);             // -2*x[5]*x[13]
    addBilinearTerm(L, row, -2.0, 3, 13, V);             // -2*x[3]*x[13]
    addBilinearTerm(L, row, -2.0, 0, 14, V);             // -2*x[0]*x[14]
    addBilinearTerm(L, row, -2.0, 7, 14, V);             // -2*x[7]*x[14]
    addBilinearTerm(L, row, +1.0, 14, 16, V);            // +1*x[14]*x[16]
    addBilinearTerm(L, row, -2.0, 8, 15, V);             // -2*x[8]*x[15]
    addBilinearTerm(L, row, -2.0, 1, 15, V);             // -2*x[1]*x[15]
    row++;
    
    // --- Constraint E5: for i = 2 ---
    // E5 = x[12] - { 2*(x[6]-x[2])*x[13] + 2*(x[8]-x[1])*x[14] + [2*(x[0]+x[9]) - x[16]]*x[15] } = 0.
    addBilinearTerm(L, row, 1.0, 12, 16, V);            // x[12]*x[16]
    addBilinearTerm(L, row, -2.0, 6, 13, V);             // -2*x[6]*x[13]
    addBilinearTerm(L, row, +2.0, 2, 13, V);             // +2*x[2]*x[13]
    addBilinearTerm(L, row, -2.0, 8, 14, V);             // -2*x[8]*x[14]
    addBilinearTerm(L, row, +2.0, 1, 14, V);             // +2*x[1]*x[14]
    addBilinearTerm(L, row, -2.0, 0, 15, V);             // -2*x[0]*x[15]
    addBilinearTerm(L, row, -2.0, 9, 15, V);             // -2*x[9]*x[15]
    addBilinearTerm(L, row, +1.0, 15, 16, V);            // +1*x[15]*x[16]
    row++;
    
    // ---------- (c) Scaling Constraint (1 equation) ----------
    // Enforce ρ² - 1 = 0. (ρ is x[16].)
    addBilinearTerm(L, row++, 1.0, 16, 16, V);  // This adds x[16]*x[16]
    // Then subtract the constant “1”. In the homogeneous formulation, we
    // represent the constant 1 as 1*ρ². Hence, we add -1 times that term:
    // (Since our system is homogeneous, this constraint forces ρ to be 1 up to scale.)
    // For simplicity we incorporate this by subtracting 1 from the diagonal term.
    L(row-1, 0) -= 1.0;  // (Alternatively, one could add an extra column representing the constant.)
    
    if (row != 27) {
        throw std::runtime_error("Internal error: Expected 27 rows in L, but got a different number.");
    }
    
    return L;
}



// Step 2: Enforce the quadratic consistency constraints
// Suppose the nullspace of M_full has dimension N (N >= 1) and let
// {μ₁, …, μ_N} be a basis (the columns of V). Then every solution can be written as:
//    x̄ = Σᵢ₌₁ᴺ λᵢ μᵢ
// The 27 quadratic constraints (20 from the quaternion monomials, 6 from the p₀–r₀ relation,
// and 1 from the scaling) must hold for x̄. Writing these constraints in terms of the λ’s
// yields a homogeneous system in the monomials λᵢλⱼ. Let λ̄ be the vector of all products λᵢλⱼ,
// of length N(N+1)/2. Then we have
//    L · λ̄ = 0   (44)
// where L is a 27×[N(N+1)/2] matrix whose rows are linear functions of the products λᵢλⱼ.
// In what follows, we write a function that “builds” L and solves for λ̄, then recovers λ.
inline Eigen::VectorXd enforceQuadraticConstraints(const Eigen::MatrixXd &V) {
    int N = V.cols();  // dimension of the nullspace
    int dimLambda = N * (N + 1) / 2;  // number of monomials in λᵢλⱼ
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(27, dimLambda);
    
    // --- Build the 27 constraints ---
    // (i) Twenty constraints from the consistency of the 10 quaternion monomials.
    //     For example, one constraint is: (q₁q₂)² - (q₁²)(q₂²) = 0.
    //     In terms of the lifted solution, if the quaternion monomials are given by the first 10 entries
    //     of x̄, then for the corresponding coefficients in each μᵢ (i=1..N) you can write:
    //         μᵢ(index corresponding to q₁q₂)
    //         and μᵢ(index corresponding to q₁²) and μᵢ(index corresponding to q₂²)
    //     Then the constraint becomes a linear equation in the products λᵢλⱼ.
    // (ii) Six constraints arise from the relation between p₀ and r₀ (recall r₀ = C₀ᵀp₀).
    // (iii) One constraint enforces the scaling (ρ = 1).

    L = computeLMatrix(V); 
    
    // Solve L * λ̄ = 0.
    Eigen::JacobiSVD<Eigen::MatrixXd> svdL(L, Eigen::ComputeFullV);
    Eigen::MatrixXd VL = svdL.matrixV();
    int dim = VL.cols();
    // The solution λ̄ is the singular vector corresponding to the smallest singular value.
    Eigen::VectorXd lambda_bar = VL.col(dim - 1);
    
    // Recover the individual λ coefficients.
    // For each i, the (i,i) entry in the product vector corresponds to λᵢ².
    Eigen::VectorXd lambda(N);
    for (int i = 0; i < N; ++i) {
        int idx = i * (i + 1) / 2;  // index corresponding to λᵢ² in λ̄
        lambda(i) = std::sqrt(std::abs(lambda_bar(idx)));  // choose the positive square root
    }
    return lambda;
}


// -------------------------------------------------------------------------
// Step 3: Reconstruct x̄ using the lambda coefficients and the nullspace basis.
// That is, compute x̄ = Σᵢ₌₁ᴺ λᵢ μᵢ.
inline Eigen::VectorXd reconstructLiftedSolution(const Eigen::MatrixXd &V, const Eigen::VectorXd &lambda) {
    int N = V.cols();
    int cols = V.rows();  // should be 17
    Eigen::VectorXd xbar = Eigen::VectorXd::Zero(cols);
    for (int i = 0; i < N; ++i) {
        xbar += lambda(i) * V.col(i);
    }
    return xbar;
}


// This function computes an algebraic initialization for the distance-only relative
// transformation between the robot odometry frames. It uses measurements taken between
// anchors (mounted on the AGV body) and tags (mounted on the UAV body).
inline Sophus::SE3d algebraicDistanceInitialization(const std::vector<Measurement> &subset) {

    const size_t minMeasurements = 10;
    if (subset.size() < minMeasurements) {
        throw std::runtime_error("At least 10 measurements are required for algebraic initialization.");
    }
    
    Eigen::MatrixXd M_full = buildLiftedMatrix(subset);
    
    // Step 2: Solve for the nullspace of M. In the ideal (noise-free) case, the nullspace is one-dimensional.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_full, Eigen::ComputeFullV);
    Eigen::VectorXd singularValues = svd.singularValues();
    int cols = M_full.cols();  // 17
    // Determine the numerical rank (using a tolerance) to obtain the nullspace dimension N.
    double tol = 1e-6;
    int N = 0;
    for (int i = 0; i < cols; ++i) {
        if (singularValues(i) < tol) N++;
    }
    if (N < 1) {
        throw std::runtime_error("No nullspace found; the measurements may be inconsistent.");
    }
    
    // Let V be a 17×N matrix whose columns form an orthonormal basis for the nullspace.
    Eigen::MatrixXd V = svd.matrixV().rightCols(N);
    
    // If the nullspace dimension N > 1, enforce the 27 quadratic consistency constraints.
    Eigen::VectorXd lambda;
    if (N > 1) {
        lambda = enforceQuadraticConstraints(V);
    } else {
        lambda = Eigen::VectorXd::Ones(1);
    }
    
    // Reconstruct the lifted solution x̄ = Σᵢ λᵢ μᵢ.
    Eigen::VectorXd xbar = reconstructLiftedSolution(V, lambda);
    
    // Recover the quaternion from the first 10 entries of x̄.
    Eigen::VectorXd qMonomials = xbar.head(10);
    Eigen::Vector4d q = recoverQuaternionFromMonomials(qMonomials);
    q.normalize();
    
    // Recover the translation p₀ from entries 10 to 12.
    Eigen::Vector3d p0 = xbar.segment<3>(10);
    
    Sophus::SO3d R(q);
    Sophus::SE3d T_relative(R, p0);
    return T_relative;
}


} // namespace uwb_localization

#endif  // UWB_LOCALIZATION_ALGEBRAIC_INITIALIZATION_H_