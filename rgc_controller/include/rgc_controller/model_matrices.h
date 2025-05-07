#ifndef MODEL_MATRICES_H
#define MODEL_MATRICES_H

#include <OsqpEigen/OsqpEigen.h>
#include "rgc_controller/model_matrices.h"
#include <cmath>
#include <vector>

class ModelMatrices
{
private:
public:
    ModelMatrices();

    ~ModelMatrices();

    Eigen::Matrix3d InitializeInertiaTensor(double m, double r, double h);

    void InitializeHTMatrices();

    void ComputeSinCos();

    void UpdateRobotStates(const Eigen::Matrix<double, 3, 1> &_q, const Eigen::Matrix<double, 3, 1> &_dq, double th,
                           const Eigen::Matrix<double, 2, 1> &_b, const Eigen::Matrix<double, 2, 1> &_db);

    void UpdateHomogTrans();

    void UpdateJacobians();

    void UpdateRotMtx(double th);

    void CoMJacobian();

    void ComputeMassMatrix();

    void ComputeCoriolisMatrix();
    // Eigen::Matrix3d CoMJacobian();

    // Eigen::Vector3d CoMPos();
    void CoMPos();

    Eigen::Matrix3d InertiaTensor();

    void ComputeIq();

    double L0, L1, L2, L3, r;
    double m0, m1, m2, m3, m;

    std::vector<double> masses;
    std::vector<Eigen::Matrix3d> Inertia;

    Eigen::Matrix3d I0_inv, M, C, J_com_rot, J_com_rot_constant;

    // Transformation Matrices
    Eigen::Matrix4d HT_com0, HT_com1, HT_com2, HT_com3, HT_toe, HT_heel, HT_knee, HT_ankle;

    std::vector<Eigen::Matrix4d *> HT;

    //
    Eigen::Matrix3d Rot_mtx;

    // Jacobians
    Eigen::Matrix3d J_com, J_com1, J_com2, J_com3, J_toe, J_heel, J_ankle, J_rot_matrix;

    // Robot states
    Eigen::Vector3d q, CoM, dq; // Joint positions

    // Precomputed sine and cosine values
    Eigen::Vector3d sin_vals, cos_vals;

    Eigen::Matrix<double, 3, 1> qU, qL, tau_lim;
    Eigen::Matrix<double, 1, 3> Iq;
    Eigen::Matrix<double, 2, 1> b, db;

    std::vector<Eigen::Matrix<double, 3, 3> *> J_com_ptr;
};
#endif