#ifndef UWB_LOCALIZATION_MANIFOLDS_H_
#define UWB_LOCALIZATION_MANIFOLDS_H_

#include <ceres/ceres.h>
#include <cmath>

namespace uwb_localization
{
    // Custom manifold for a state in R^3 x S^1.
    class StateManifold4D : public ceres::Manifold {
        public:
            // Ambient space dimension is 4.
            virtual int AmbientSize() const override { return 4; }
            // Tangent space dimension is also 4.
            virtual int TangentSize() const override { return 4; }
            
            // Plus(x, delta) = [x0+delta0, x1+delta1, x2+delta2, wrap(x3+delta3)]
            virtual bool Plus(const double* x,
                                const double* delta,
                                double* x_plus_delta) const override {
                // Translation is standard addition.
                x_plus_delta[0] = x[0] + delta[0];
                x_plus_delta[1] = x[1] + delta[1];
                x_plus_delta[2] = x[2] + delta[2];
                // For yaw, perform addition with wrapping to [-pi, pi].
                double new_yaw = x[3] + delta[3];
                while(new_yaw > M_PI)  new_yaw -= 2.0 * M_PI;
                while(new_yaw < -M_PI) new_yaw += 2.0 * M_PI;
                x_plus_delta[3] = new_yaw;
                return true;
            }
            
            // The Jacobian of Plus with respect to delta at delta = 0 is the identity.
            virtual bool PlusJacobian(const double* /*x*/, double* jacobian) const override {
                // Fill a 4x4 identity matrix (row-major ordering).
                for (int i = 0; i < 16; ++i) {
                jacobian[i] = 0.0;
                }
                jacobian[0]  = 1.0;
                jacobian[5]  = 1.0;
                jacobian[10] = 1.0;
                jacobian[15] = 1.0;
                return true;
            }
            
            // Minus(y, x) computes the tangent vector delta such that x minus delta = y.
            virtual bool Minus(const double* y,
                                const double* x,
                                double* y_minus_x) const override {
                // For translation, simple subtraction.
                y_minus_x[0] = y[0] - x[0];
                y_minus_x[1] = y[1] - x[1];
                y_minus_x[2] = y[2] - x[2];
                // For yaw, compute the difference and wrap it.
                double dtheta = y[3] - x[3];
                while (dtheta > M_PI)  dtheta -= 2.0 * M_PI;
                while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
                y_minus_x[3] = dtheta;
                return true;
            }
            
            // The Jacobian of Minus with respect to y at y = x is the identity.
            virtual bool MinusJacobian(const double* /*x*/, double* jacobian) const override {
                for (int i = 0; i < 16; ++i) {
                jacobian[i] = 0.0;
                }
                jacobian[0]  = 1.0;
                jacobian[5]  = 1.0;
                jacobian[10] = 1.0;
                jacobian[15] = 1.0;
                return true;
            }
    };

    class StateManifold3D : public ceres::Manifold {
        public:
          // Ambient dimension is 4 
          int AmbientSize() const override { return 4; }
          // Tangent dimension is 3 
          int TangentSize() const override { return 3; }
        
          // x ⊕ δ  →  writes 4 components
          bool Plus(const double* x,
                    const double* delta,
                    double* x_plus_delta) const override {
            // translation
            x_plus_delta[0] = x[0] + delta[0];  // x
            x_plus_delta[1] = x[1] + delta[1];  // y
            x_plus_delta[2] = x[2];             // z stays fixed
            // yaw with wrapping
            double yaw = x[3] + delta[2];
            while (yaw > M_PI)  yaw -= 2.0 * M_PI;
            while (yaw < -M_PI) yaw += 2.0 * M_PI;
            x_plus_delta[3] = yaw;
            return true;
          }
        
          // ∂(x⊕δ)/∂δ  is 4 rows × 3 cols
          bool PlusJacobian(const double* /*x*/,
                            double* jacobian) const override {
            // zero 4×3
            std::fill(jacobian, jacobian + 4*3, 0.0);
            // ∂x₀/∂δ₀ = 1
            jacobian[0*3 + 0] = 1.0;
            // ∂x₁/∂δ₁ = 1
            jacobian[1*3 + 1] = 1.0;
            // ∂x₃/∂δ₂ = 1  (yaw)
            jacobian[3*3 + 2] = 1.0;
            return true;
          }
        
          //writes 3 components
          bool Minus(const double* y,
                     const double* x,
                     double* y_minus_x) const override {
            y_minus_x[0] = y[0] - x[0];  // Δx
            y_minus_x[1] = y[1] - x[1];  // Δy
            // Δyaw with wrapping
            double d = y[3] - x[3];
            while (d > M_PI)  d -= 2.0 * M_PI;
            while (d < -M_PI) d += 2.0 * M_PI;
            y_minus_x[2] = d;
            return true;
          }
        
          // 3 rows × 4 cols
          bool MinusJacobian(const double* /*x*/,
                             double* jacobian) const override {
            // zero 3×4
            std::fill(jacobian, jacobian + 3*4, 0.0);
            // ∂Δx/∂y₀ = 1
            jacobian[0*4 + 0] = 1.0;
            // ∂Δy/∂y₁ = 1
            jacobian[1*4 + 1] = 1.0;
            // ∂Δyaw/∂y₃ = 1
            jacobian[2*4 + 3] = 1.0;
            return true;
          }
        };

}

#endif