#include "rgc_controller/model_matrices.h"

ModelMatrices::ModelMatrices()
{
    // Initialize constants
    L0 = 0.4;
    L1 = 0.45;
    L2 = 0.5;
    L3 = 0.39;
    r = 0.05;
    m0 = 7;
    m1 = 1;
    m2 = 1;
    m3 = 1;

    masses = {m0, m1, m2, m3};
    m = m0 + m1 + m2 + m3;

    // Initialize inertia tensors
    Inertia.push_back(this->InitializeInertiaTensor(m0, r, L0));
    Inertia.push_back(this->InitializeInertiaTensor(m1, r, L1));
    Inertia.push_back(this->InitializeInertiaTensor(m2, r, L2));
    Inertia.push_back(this->InitializeInertiaTensor(m3, r, L3));

    // Apply rotation to I3
    Eigen::Matrix3d arr;
    arr << 0, 0, 1,
        0, 1, 0,
        -1, 0, 0;
    Inertia[3] = arr * Inertia[3] * arr.transpose();

    J_rot_matrix.setZero();
    J_rot_matrix << 1, 0, 0, // Link 1
        1, 1, 0,             // Link 2
        1, 1, 1;             // Link 3

    // Initialize Jacobians and HT matrices with zeros
    J_com.setZero();
    J_com1.setZero();
    J_com2.setZero();
    J_com3.setZero();
    J_toe.setZero();
    J_heel.setZero();
    J_ankle.setZero();
    HT_com0.setZero();
    HT_com1.setZero();
    HT_com2.setZero();
    HT_com3.setZero();
    HT_knee.setZero();
    HT_ankle.setZero();
    HT_toe.setZero();
    HT_heel.setZero();
    Rot_mtx.setZero();

    // Transformation matrices list
    HT = {&HT_com0, &HT_com1, &HT_com2, &HT_com3};

    // Initialize constants in HT matrices
    InitializeHTMatrices();

    // Initialize state and precomputed values
    q.setZero();
    dq.setZero();
    sin_vals.setZero();
    cos_vals.setZero();

    Rot_mtx(1, 1) = 1;

    this->qL << -1.40, -0.5, -1.1;
    this->qU << 0.50, 2.2, 1.1;

    // this->qL << -0.5, -2.2, -1.1;
    // this->qU << 1.4, 0.5, 1.1;

    J_com_ptr = {&J_com1, &J_com2, &J_com3};

    tau_lim << 50, 50, 50;
}

// Destructor
ModelMatrices::~ModelMatrices() {}

Eigen::Matrix3d ModelMatrices::InitializeInertiaTensor(double m, double r, double h)
{
    Eigen::Matrix3d Inertia_tensor = Eigen::Matrix3d::Zero();
    Inertia_tensor(0, 0) = ((m * h * h) / 12.0) + ((m * r * r) / 4.0);
    Inertia_tensor(1, 1) = Inertia_tensor(0, 0);
    Inertia_tensor(2, 2) = (m * r * r) / 2.0;
    return Inertia_tensor;
}

void ModelMatrices::InitializeHTMatrices()
{
    HT_com1(3, 3) = 1.0;
    HT_com1(1, 2) = -1.0;
    HT_com2(3, 3) = 1.0;
    HT_com2(1, 2) = -1.0;
    HT_com3(3, 3) = 1.0;
    HT_com3(1, 2) = -1.0;

    HT_toe(3, 3) = 1.0;
    HT_toe(1, 2) = -1.0;
    HT_heel(3, 3) = 1.0;
    HT_heel(1, 2) = -1.0;
    HT_knee(3, 3) = 1.0;
    HT_knee(1, 2) = -1.0;
    HT_ankle(3, 3) = 1.0;
    HT_ankle(1, 2) = -1.0;

    HT_com0.setIdentity(); // Identity matrix for HT_com0
    HT_com0(2, 3) = L0 / 2.0;
}

void ModelMatrices::ComputeSinCos()
{
    sin_vals(0) = sin(q(0));
    cos_vals(0) = cos(q(0));
    sin_vals(1) = sin(q(0) + q(1));
    cos_vals(1) = cos(q(0) + q(1));
    sin_vals(2) = sin(q(0) + q(1) + q(2));
    cos_vals(2) = cos(q(0) + q(1) + q(2));
}

void ModelMatrices::UpdateRobotStates(const Eigen::Matrix<double, 3, 1> &_q, const Eigen::Matrix<double, 3, 1> &_dq, double th,
                                      const Eigen::Matrix<double, 2, 1> &_b, const Eigen::Matrix<double, 2, 1> &_db)

{
    q = _q;
    dq = _dq;
    b = _b;
    db = _db;
    ComputeSinCos(); // Recompute sine and cosine values
    UpdateRotMtx(th);
    UpdateHomogTrans();
    UpdateJacobians();
}

void ModelMatrices::UpdateRotMtx(double th)
{
    Rot_mtx(0, 0) = cos(th);
    Rot_mtx(0, 2) = sin(th);
    Rot_mtx(2, 0) = -Rot_mtx(0, 2);
    Rot_mtx(2, 2) = Rot_mtx(0, 0);
}

void ModelMatrices::UpdateHomogTrans()
{
    // Update HT_com1
    HT_com1(0, 0) = sin_vals(0);
    HT_com1(0, 1) = cos_vals(0);
    HT_com1(0, 3) = 0.225 * sin_vals(0);
    HT_com1(2, 0) = -cos_vals(0);
    HT_com1(2, 1) = sin_vals(0);
    HT_com1(2, 3) = -0.225 * cos_vals(0);

    // Update HT_com2
    HT_com2(0, 0) = sin_vals(1);
    HT_com2(0, 1) = cos_vals(1);
    HT_com2(0, 3) = 0.45 * sin_vals(0) + 0.25 * sin_vals(1);
    HT_com2(2, 0) = -cos_vals(1);
    HT_com2(2, 1) = sin_vals(1);
    HT_com2(2, 3) = -0.45 * cos_vals(0) - 0.25 * cos_vals(1);

    // Update HT_com3
    HT_com3(0, 0) = sin_vals(2);
    HT_com3(0, 1) = cos_vals(2);
    HT_com3(0, 3) = 0.45 * sin_vals(0) + 0.065 * cos_vals(2) + 0.5 * sin_vals(1);
    HT_com3(2, 0) = -cos_vals(2);
    HT_com3(2, 1) = sin_vals(2);
    HT_com3(2, 3) = 0.065 * sin_vals(2) - 0.45 * cos_vals(0) - 0.5 * cos_vals(1);

    // Update HT_toe
    HT_toe(0, 0) = sin_vals(2);
    HT_toe(0, 1) = cos_vals(2);
    HT_toe(0, 3) = 0.45 * sin_vals(0) + 0.26 * cos_vals(2) + 0.05 * sin_vals(2) + 0.50 * sin_vals(1);
    HT_toe(2, 0) = -cos_vals(2);
    HT_toe(2, 1) = sin_vals(2);
    HT_toe(2, 3) = 0.26 * sin_vals(2) - 0.05 * cos_vals(2) - 0.45 * cos_vals(0) - 0.5 * cos_vals(1);

    // Update HT_heel
    HT_heel(0, 0) = sin_vals(2);
    HT_heel(0, 1) = cos_vals(2);
    HT_heel(0, 3) = 0.45 * sin_vals(0) - 0.13 * cos_vals(2) + 0.05 * sin_vals(2) + 0.50 * sin_vals(1);
    HT_heel(2, 0) = -cos_vals(2);
    HT_heel(2, 1) = sin_vals(2);
    HT_heel(2, 3) = -0.13 * sin_vals(2) - 0.05 * cos_vals(2) - 0.45 * cos_vals(0) - 0.5 * cos_vals(1);

    // Update HT_knee
    HT_knee(0, 0) = sin_vals(1);
    HT_knee(0, 1) = cos_vals(1);
    HT_knee(0, 3) = 0.45 * sin_vals(0);
    HT_knee(2, 0) = -cos_vals(1);
    HT_knee(2, 1) = sin_vals(1);
    HT_knee(2, 3) = -0.45 * cos_vals(0);

    // / Update HT_ankle
    HT_ankle(0, 0) = sin_vals(2);
    HT_ankle(0, 1) = cos_vals(2);
    HT_ankle(0, 3) = 0.45 * sin_vals(0) + 0.5 * sin_vals(1);
    HT_ankle(2, 0) = -cos_vals(2);
    HT_ankle(2, 1) = sin_vals(2);
    HT_ankle(2, 3) = -0.45 * cos_vals(0) - 0.5 * cos_vals(1);
}

void ModelMatrices::UpdateJacobians()
{
    // Jacobian for HT_toe
    J_toe(0, 0) = 0.45 * cos_vals(0) - 0.26 * sin_vals(2) + 0.05 * cos_vals(2) + 0.50 * cos_vals(1);
    J_toe(0, 1) = -0.26 * sin_vals(2) + 0.05 * cos_vals(2) + 0.50 * cos_vals(1);
    J_toe(0, 2) = -0.26 * sin_vals(2) + 0.05 * cos_vals(2);

    J_toe(2, 0) = 0.26 * cos_vals(2) + 0.05 * sin_vals(2) + 0.45 * sin_vals(0) + 0.50 * sin_vals(1);
    J_toe(2, 1) = 0.26 * cos_vals(2) + 0.05 * sin_vals(2) + 0.50 * sin_vals(1);
    J_toe(2, 2) = 0.26 * cos_vals(2) + 0.05 * sin_vals(2);

    // Jacobian for HT_heel
    J_heel(0, 0) = 0.45 * cos_vals(0) + 0.13 * sin_vals(2) + 0.05 * cos_vals(2) + 0.50 * cos_vals(1);
    J_heel(0, 1) = -0.13 * sin_vals(2) + 0.05 * cos_vals(2) + 0.50 * cos_vals(1);
    J_heel(0, 2) = -0.13 * sin_vals(2) + 0.05 * cos_vals(2);

    J_heel(2, 0) = -0.26 * cos_vals(2) + 0.05 * sin_vals(2) + 0.45 * sin_vals(0) + 0.50 * sin_vals(1);
    J_heel(2, 1) = -0.26 * cos_vals(2) + 0.05 * sin_vals(2) + 0.50 * sin_vals(1);
    J_heel(2, 2) = -0.26 * cos_vals(2) + 0.05 * sin_vals(2);

    // Jacobian for HT_ankle
    J_ankle(0, 0) = 0.45 * cos_vals(0) + 0.5 * cos_vals(1);
    J_ankle(0, 1) = 0.5 * cos_vals(1);

    J_ankle(2, 0) = 0.45 * sin_vals(0) + 0.5 * sin_vals(1);
    J_ankle(2, 1) = 0.5 * sin_vals(1);

    // Jacobian for J_com1
    J_com1(0, 0) = 0.225 * cos_vals(0);
    J_com1(2, 0) = 0.225 * sin_vals(0);

    // Jacobian for J_com2
    J_com2(0, 0) = 0.45 * cos_vals(0) + 0.25 * cos_vals(1);
    J_com2(0, 1) = 0.25 * cos_vals(1);

    J_com2(2, 0) = 0.45 * sin_vals(0) + 0.25 * sin_vals(1);
    J_com2(2, 1) = 0.25 * sin_vals(1);

    // Jacobian for J_com3
    J_com3(0, 0) = 0.45 * cos_vals(0) - 0.065 * sin_vals(2) + 0.5 * cos_vals(1);
    J_com3(0, 1) = 0.5 * cos_vals(1) - 0.065 * sin_vals(2);
    J_com3(0, 2) = -0.065 * sin_vals(2);

    J_com3(2, 0) = 0.45 * sin_vals(0) + 0.065 * cos_vals(2) + 0.5 * sin_vals(1);
    J_com3(2, 1) = 0.5 * sin_vals(1) + 0.065 * cos_vals(2);
    J_com3(2, 2) = 0.065 * cos_vals(2);
}

// Update CoM Position
void ModelMatrices::CoMPos()
{

    CoM = (HT_com0.block<3, 1>(0, 3) * m0 +
           HT_com1.block<3, 1>(0, 3) * m1 +
           HT_com2.block<3, 1>(0, 3) * m2 +
           HT_com3.block<3, 1>(0, 3) * m3) /
          m;
}

void ModelMatrices::CoMJacobian()
{
    J_com = (J_com1 * m1 + J_com2 * m2 + J_com3 * m3) / m;
}

// Update Inertia Tensor
Eigen::Matrix3d ModelMatrices::InertiaTensor()
{
    Eigen::Matrix3d Ib = Eigen::Matrix3d::Zero();
    for (size_t i = 0; i < masses.size(); ++i)
    {
        Eigen::Matrix3d rot = HT[i]->block<3, 3>(0, 0);   // Get rotation matrix
        Eigen::Vector3d trans = HT[i]->block<3, 1>(0, 3); // Get translation vector
        Ib += rot * Inertia[i] * rot.transpose() + (masses[i] * trans.squaredNorm() * Eigen::Matrix3d::Identity() - trans * trans.transpose());
    }

    return Ib;
}

void ModelMatrices::ComputeMassMatrix()
{
    // Compute missing sin and cos terms
    double sin_q1_q2 = sin(q(1) + q(2));
    double cos_q1_q2 = cos(q(1) + q(2));

    // Initialize the mass matrix
    M.setZero();

    M(0, 0) = -0.065 * sin_vals(2) - 0.0585 * sin_q1_q2 + 0.675 * cos_vals(1) + 0.006025 * cos(2 * q(0) + 2 * q(1) + 2 * q(2)) + 0.818583333333334;
    M(0, 1) = -0.065 * sin_vals(2) - 0.02925 * sin_q1_q2 + 0.3375 * cos_vals(1) + 0.316725;
    M(0, 2) = -0.0325 * sin_vals(2) - 0.02925 * sin_q1_q2 + 0.006025 * sin(2 * q(0) + 2 * q(1) + 2 * q(2)) + 0.004225;

    M(1, 0) = M(0, 1);
    M(1, 1) = 0.332525 - 0.065 * sin_vals(2);
    M(1, 2) = 0.004225 - 0.0325 * sin_vals(2);

    M(2, 0) = M(0, 2);
    M(2, 1) = M(1, 2);
    M(2, 2) = -0.006025 * cos(2 * q(0) + 2 * q(1) + 2 * q(2)) + 0.0504583333333333;
}

void ModelMatrices::ComputeCoriolisMatrix()
{
    // Compute missing sin and cos terms
    double sin_q1_q2 = sin(q(1) + q(2));
    double cos_q1_q2 = cos(q(1) + q(2));

    // Compute double-angle identities
    double sin_2q = 2 * sin_vals(2) * cos_vals(2);
    double cos_2q = 2 * cos_vals(2) * cos_vals(2) - 1;

    // Initialize the Coriolis matrix
    C.setZero();

    C(0, 0) = -0.006025 * dq(0) * sin_2q - dq(1) * (0.3375 * sin_vals(1) + 0.006025 * sin_2q + 0.02925 * cos_q1_q2) - dq(2) * (0.006025 * sin_2q + 0.0325 * cos_vals(2) + 0.02925 * cos_q1_q2);

    C(0, 1) = -dq(0) * (0.3375 * sin_vals(1) + 0.006025 * sin_2q + 0.02925 * cos_q1_q2) - dq(1) * (0.3375 * sin_vals(1) + 0.02925 * cos_q1_q2) - dq(2) * (0.0325 * cos_vals(2) + 0.02925 * cos_q1_q2 - 0.006025 * cos_2q);

    C(0, 2) = -dq(0) * (0.006025 * sin_2q + 0.0325 * cos_vals(2) + 0.02925 * cos_q1_q2) - dq(1) * (0.0325 * cos_vals(2) + 0.02925 * cos_q1_q2 - 0.006025 * cos_2q) - dq(2) * (0.006025 * sin_2q + 0.0325 * cos_vals(2) + 0.02925 * cos_q1_q2 - 0.01205 * cos_2q);

    C(1, 0) = dq(0) * (0.3375 * sin_vals(1) + 0.006025 * sin_2q + 0.02925 * cos_q1_q2) - dq(2) * (0.0325 * cos_vals(2) + 0.006025 * cos_2q);

    C(1, 1) = -0.0325 * dq(2) * cos_vals(2);

    C(1, 2) = -dq(0) * (0.0325 * cos_vals(2) + 0.006025 * cos_2q) - 0.0325 * dq(1) * cos_vals(2);

    C(2, 0) = dq(0) * (0.006025 * sin_2q + 0.0325 * cos_vals(2) + 0.02925 * cos_q1_q2 + 0.01205 * cos_2q) + dq(1) * (0.0325 * cos_vals(2) + 0.006025 * cos_2q);

    C(2, 1) = dq(0) * (0.0325 * cos_vals(2) + 0.006025 * cos_2q) + 0.0325 * dq(1) * cos_vals(2);

    C(2, 2) = dq(0) * (0.006025 * sin_2q) - dq(1) * (0.006025 * sin_2q) - dq(2) * (0.006025 * sin_2q);
}

void ModelMatrices::ComputeIq()
{
    // Reset Iq as a 1x3 row vector
    Iq.setZero();

    // Compute rotational contribution
    for (size_t i = 1; i < masses.size(); ++i)
    {
        Eigen::Vector3d ri = HT[i]->block<3, 1>(0, 3) - CoM; // Relative position to CoM
        Eigen::Matrix3d skew_ri;
        skew_ri << 0, -ri(2), ri(1),
            ri(2), 0, -ri(0),
            -ri(1), ri(0), 0;

        // Linear contribution using mass and Jacobian (1x3 row vector)
        Iq += J_rot_matrix.block(i - 1, 0, 1, 3) * Inertia[i](1, 1) + masses[i] * skew_ri.row(1) * (*J_com_ptr[i - 1]);
    }
}