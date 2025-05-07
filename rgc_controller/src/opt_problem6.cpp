#include "rgc_controller/opt_problem6.h"

OptProblem6::OptProblem6(ModelMatrices *Robot) : RobotMtx(Robot)
{
    // Resize dynamic model matrices
    this->A.resize(10, 10);
    this->A.setZero();

    this->B.resize(10, 3);
    this->B.setZero();

    this->Aa.resize(13, 13);
    this->Aa.setZero();

    this->Ba.resize(13, 3);
    this->Ba.setZero();

    this->Ca.resize(3, 13);
    this->Ca.setZero();

    // q, tau
    this->C_cons.resize(6, 13);
    this->C_cons.setZero();

    this->C_consV.resize(this->C_cons.rows(), 1);

    this->ref.resize(3, 1);
    this->ref.setZero();
}

OptProblem6::~OptProblem6()
{
}

void OptProblem6::UpdateModelConstants()
{

    /*
    x = |dq q dth th r|   u = qr

    M^{-1}=iM
    Iq^{-1}*I0 = Iyy

    A = | -iM*(Kd+C)   -iM*kp   0 0 0| B = |iM*kp      |
        |       I           0   0 0 0|     |     0     |
        | Iyy*iM*kd  Iyy*iM*kp  0 0 0|     |-Iyy*iM*kp |
        |       I           0   0 0 0|     |     0     |
        | J_com             0   0 0 0|     |     0     |

    x = |dq q dth th r qra|   u = dqr

    Aa = | Ad  Bd|  Ba = | Bd|
         | 0   I |       | I |

    Ca = |0 I 0 0 0| y = |q|
*/

    // 60 -100 50

    // this->ref << -60, 100, -40;
    // this->ref << 78, -120, 50;

    this->ref << -70, 120, -50;
    this->ref = this->ref * PI / 180;
    // Initialize matrices constants

    this->A.block(3, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    this->A(7, 0) = 1;

    this->Ca.block(0, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3);

    this->Aa.block(10, 10, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    this->Ba.block(10, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);

    // Initializa constraints matrix

    this->C_cons.block(0, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    this->C_cons.block(3, 0, 3, 3) = -Kd * Eigen::MatrixXd::Identity(3, 3);
    this->C_cons.block(3, 3, 3, 3) = -Kp * Eigen::MatrixXd::Identity(3, 3);
    this->C_cons.block(3, 10, 3, 3) = Kp * Eigen::MatrixXd::Identity(3, 3);

    this->C_consV.setZero();
    this->Ucv_var.setZero();
    this->Lcv_var.setZero();

    Eigen::MatrixXd Ub, Lb;
    Ub.resize(this->C_cons.rows(), 1);
    Ub << RobotMtx->qU, RobotMtx->tau_lim;
    Lb.resize(this->C_cons.rows(), 1);
    Lb << RobotMtx->qL, -RobotMtx->tau_lim;

    this->UpdateReferences(this->ref);
    this->SetConsBounds(Lb, Ub);
}

void OptProblem6::UpdateDynamicModel()
{

    /*
    x = |dq q dth th r|   u = qr

    M^{-1}=iM
    Iq^{-1}*I0 = Iyy

    A = | -iM*(Kd+C)   -iM*kp   0 0 0| B = |iM*kp      |
        |       I           0   0 0 0|     |     0     |
        | Iyy*iM*kd  Iyy*iM*kp  0 0 0|     |-Iyy*iM*kp |
        |       I           0   0 0 0|     |     0     |
        | J_com             0   0 0 0|     |     0     |

    x = |dq q dth th r qra|   u = dqr

    Aa = | Ad  Bd|  Ba = | Bd|
         | 0   I |       | I |

    Ca = |0 I 0 0 0| y = |q|
*/
    auto roty = RobotMtx->Rot_mtx;

    RobotMtx->CoMJacobian();
    RobotMtx->CoMPos();
    RobotMtx->ComputeMassMatrix();
    RobotMtx->ComputeCoriolisMatrix();
    RobotMtx->ComputeIq();

    auto Jcom = roty * RobotMtx->J_com;
    auto inv_M = RobotMtx->M.inverse();
    auto Cor_mtx = RobotMtx->C;
    auto Iq = RobotMtx->Iq;
    auto Ib = RobotMtx->Inertia[0](1, 1);

    Eigen::Matrix<double, 2, 3> Jcs;

    Jcs << Jcom(0, 0), Jcom(0, 1), Jcom(0, 2), Jcom(2, 0), Jcom(2, 1), Jcom(2, 2); // ok
    this->A.block(0, 0, 3, 3) = -inv_M * (Cor_mtx + Kd * Eigen::MatrixXd::Identity(3, 3));
    this->A.block(0, 3, 3, 3) = -inv_M * Kp;

    this->A.block(6, 0, 1, 3) = -Iq * this->A.block(0, 0, 3, 3) / Ib;
    this->A.block(6, 3, 1, 3) = -Iq * this->A.block(0, 3, 3, 3) / Ib;

    this->A.block(8, 0, 2, 3) = Jcs;

    this->B.block(0, 0, 3, 3) = inv_M * Kp;
    this->B.block(6, 0, 1, 3) = -Iq * this->B.block(0, 0, 3, 3) / Ib;

    Aa.block(0, 0, 10, 10) = Eigen::MatrixXd::Identity(10, 10) + ts * this->A;
    Aa.block(0, 10, 10, 3) = ts * B;
    Ba.block(0, 0, 10, 3) = ts * B;
}

void OptProblem6::DefineConstraintMtxs()
{
    this->Phi_cons.block(0, 0, 3, this->nxa) = this->C_cons.block(0, 0, 3, this->nxa) * this->Aa;
    this->Phi_cons.block(3, 0, 3, this->nxa) = this->C_cons.block(3, 0, 3, this->nxa);

    this->aux_cons.block(0, 0, 3, this->nu) = this->C_cons.block(0, 0, 3, this->nxa) * this->Ba;
    this->aux_cons.block(3, 0, 3, this->nu) = Kp * Eigen::MatrixXd::Identity(3, 3);
}