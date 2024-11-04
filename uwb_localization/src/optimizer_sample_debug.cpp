#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <vector>

struct UWBResidual {
    UWBResidual(const Eigen::Vector3d& anchor, const Eigen::Vector3d& tag, double measured_distance)
        : anchor_(anchor), tag_(tag), measured_distance_(measured_distance) {}

    template <typename T>
    bool operator()(const T* const q, const T* const t, T* residual) const {
        // Rotate the anchor point
        T anchor_transformed[3];
        ceres::QuaternionRotatePoint(q, anchor_.data(), anchor_transformed);

        // Apply translation
        anchor_transformed[0] += t[0];
        anchor_transformed[1] += t[1];
        anchor_transformed[2] += t[2];

        // Calculate the vector from the transformed anchor to the tag
        T dx = T(tag_(0)) - anchor_transformed[0];
        T dy = T(tag_(1)) - anchor_transformed[1];
        T dz = T(tag_(2)) - anchor_transformed[2];
        T predicted_distance = ceres::sqrt(dx * dx + dy * dy + dz * dz);

        // Residual as (measured - predicted)
        residual[0] = T(measured_distance_) - predicted_distance;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& anchor, const Eigen::Vector3d& tag, double measured_distance) {
        return (new ceres::AutoDiffCostFunction<UWBResidual, 1, 4, 3>(
            new UWBResidual(anchor, tag, measured_distance)));
    }

    const Eigen::Vector3d anchor_;
    const Eigen::Vector3d tag_;
    const double measured_distance_;
};


int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);

    // Example data (replace with actual data)
    std::vector<Eigen::Vector3d> anchors = { /* anchor positions */ };
    std::vector<Eigen::Vector3d> tags = { /* tag positions */ };
    std::vector<std::vector<double>> distances = { /* measured distances */ };

    // Initial guess for the quaternion (no rotation) and translation (zero)
    double q[4] = {1.0, 0.0, 0.0, 0.0};  // Quaternion representing no rotation
    double t[3] = {0.0, 0.0, 0.0};       // Translation vector

    ceres::Problem problem;

    // Add residual blocks for each distance measurement
    for (size_t i = 0; i < tags.size(); ++i) {
        for (size_t j = 0; j < anchors.size(); ++j) {
            ceres::CostFunction* cost_function =
                UWBResidual::Create(anchors[j], tags[i], distances[i][j]);
            problem.AddResidualBlock(cost_function, nullptr, q, t);
        }
    }

    // Set the quaternion as a parameter block with a quaternion parameterization
    problem.SetParameterization(q, new ceres::QuaternionParameterization());

    // Configure solver options
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    // Solve
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // Output the optimized values
    std::cout << "Optimized quaternion: " << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << "\n";
    std::cout << "Optimized translation: " << t[0] << ", " << t[1] << ", " << t[2] << "\n";

    return 0;
}
