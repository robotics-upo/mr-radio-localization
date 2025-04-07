#ifndef UWB_LOCALIZATION_MANIFOLDS_H_
#define UWB_LOCALIZATION_MANIFOLDS_H_

#include <ceres/ceres.h>
#include <cmath>

namespace uwb_localization
{
    // Custom manifold for a state in R^3 x S^1.
    class StateManifold : public ceres::Manifold {
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
            
            // Minus(y, x) computes the tangent vector delta such that x ⊕ delta = y.
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

        // Custom manifold for a state in R^3 x S^1.
    class StateManifold3D : public ceres::Manifold {

        public:

            // A pointer to the flag maintained by the node.
            bool* pitch_ok_ptr_;

            // Constructor takes a pointer to the flag.
            explicit StateManifold3D(bool* pitch_ok_ptr) : pitch_ok_ptr_(pitch_ok_ptr) {}

            // Ambient space dimension is 4.
            virtual int AmbientSize() const override { return 4; }
            // Tangent space dimension is also 4.
            virtual int TangentSize() const override { return 4; }
            
            // Plus(x, delta) = [x0+delta0, x1+delta1, x2+delta2, wrap(x3+delta3)]
            virtual bool Plus(const double* x,
                                const double* delta,
                                double* x_plus_delta) const override {

                bool pitch_ok = *pitch_ok_ptr_;
                double scale = 1e-6;
                // Translation is standard addition.
                x_plus_delta[0] = x[0] + delta[0];
                x_plus_delta[1] = x[1] + delta[1];
                x_plus_delta[2] = pitch_ok ? (x[2] + delta[2]) : (x[2] + scale * delta[2]);
                // For yaw, perform addition with wrapping to [-pi, pi].
                double new_yaw = x[3] + delta[3];
                while(new_yaw > M_PI)  new_yaw -= 2.0 * M_PI;
                while(new_yaw < -M_PI) new_yaw += 2.0 * M_PI;
                x_plus_delta[3] = new_yaw;
                return true;
            }
            
            // The Jacobian of Plus with respect to delta at delta = 0 is the identity.
            virtual bool PlusJacobian(const double* /*x*/, double* jacobian) const override {

                bool pitch_ok = *pitch_ok_ptr_;
                double scale = 1e-6;
                // Fill a 4x4 identity matrix (row-major ordering).
                for (int i = 0; i < 16; ++i) {
                jacobian[i] = 0.0;
                }
                jacobian[0]  = 1.0;
                jacobian[5]  = 1.0;
                jacobian[10] = pitch_ok ? 1.0 : scale * 1.0;
                jacobian[15] = 1.0;
                return true;
            }
            
            // Minus(y, x) computes the tangent vector delta such that x ⊕ delta = y.
            virtual bool Minus(const double* y,
                                const double* x,
                                double* y_minus_x) const override {

                bool pitch_ok = *pitch_ok_ptr_;
                double scale = 1e-6;
                // For translation, simple subtraction.
                y_minus_x[0] = y[0] - x[0];
                y_minus_x[1] = y[1] - x[1];
                y_minus_x[2] = pitch_ok ? (y[2] - x[2]) : (y[2] - scale*x[2]);
                // For yaw, compute the difference and wrap it.
                double dtheta = y[3] - x[3];
                while (dtheta > M_PI)  dtheta -= 2.0 * M_PI;
                while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
                y_minus_x[3] = dtheta;
                return true;
            }
            
            // The Jacobian of Minus with respect to y at y = x is the identity.
            virtual bool MinusJacobian(const double* /*x*/, double* jacobian) const override {

                bool pitch_ok = *pitch_ok_ptr_;
                double scale = 1e-6;

                for (int i = 0; i < 16; ++i) {
                jacobian[i] = 0.0;
                }
                jacobian[0]  = 1.0;
                jacobian[5]  = 1.0;
                jacobian[10] = pitch_ok ? 1.0 : scale * 1.0;
                jacobian[15] = 1.0;
                return true;
            }
    };

}

#endif