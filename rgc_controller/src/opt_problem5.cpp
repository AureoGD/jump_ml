#include "rgc_controller/opt_problem5.h"

OptProblem5::OptProblem5(ModelMatrices *Robot) : RobotMtx(Robot)
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
    this->Ca.resize(2, 13);
    this->Ca.setZero();

    // q, tau
    this->C_cons.resize(6, 13);

    this->C_cons.setZero();

    this->C_consV.resize(this->C_cons.rows(), 1);
    this->C_consV.setZero();

    // main reference
    this->ref.resize(this->Ca.rows(), 1);
    this->ref.setZero();
}

OptProblem5::~OptProblem5()
{
}

void OptProblem5::UpdateModelConstants()
{

    /*
    xi = |r th|

    x = | dxi q xi -g|   u = qr

    A = | -Sf*T1 kp*Sf*Jcit  0 0;1| B = |-kp*Sf*Jcint|
        | -Sm*T1 kp*Sm*Jcit  0  0 |     |-kp*Sd*Jcint|
        |    T0      0       0  0 |     |     0      |
        |    I       0       0  0 |     |     0      |
        |    0       0       0  0 |     |     0      |

    xa = | dxi q xi -g qra|   u = dqr

    Aa = | Ad  Bd|  Ba = | Bd|
         | 0   I |       | I |

    Ca = |0 I 0 0 0| y = |q|
*/
    // this->ref << 20, -15, 20;
    // this->ref = this->ref * PI / 180;

    // Initialize matrices constants

    this->A(1, 9) = 1.0;
    this->A.block(6, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);

    // this->Ca.block(0, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    this->Ca(0, 7) = 1;
    this->Ca(1, 8) = 1;
    // this->Ca(0, 1) = 1.0;

    this->Aa.block(10, 10, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    this->Ba.block(10, 0, 3, 3) = Eigen::MatrixXd::Identity(3, 3);

    this->sum_f.resize(2, 3);
    this->sum_f.setZero();
    this->sum_f.block(0, 0, 2, 2) = Eigen::MatrixXd::Identity(2, 2) / RobotMtx->m;

    this->sum_m.resize(1, 3);
    this->sum_m.setZero();

    // Initializa constraints matrix

    this->C_cons.block(0, 3, 3, 3) = Eigen::MatrixXd::Identity(3, 3);
    this->C_cons.block(3, 3, 3, 3) = -Kp * Eigen::MatrixXd::Identity(3, 3);
    this->C_cons.block(3, 10, 3, 3) = Kp * Eigen::MatrixXd::Identity(3, 3);

    Eigen::MatrixXd Ub, Lb;

    Ub.resize(this->C_cons.rows(), 1);
    Ub << RobotMtx->qU, RobotMtx->tau_lim;
    Lb.resize(this->C_cons.rows(), 1);
    Lb << RobotMtx->qL, -RobotMtx->tau_lim;
    // this->ref(0, 0) = 0;
    this->ref(0, 0) = 2;
    this->ref(1, 0) = 0.3;
    this->UpdateReferences(this->ref);
    this->SetConsBounds(Lb, Ub);
}

void OptProblem5::UpdateDynamicModel()
{

    /*
    xi = |r th|

    x = | dxi q xi -g|   u = qr

    A = | -Sf*T1 kp*Sf*Jcit  0 0;1| B = |-kp*Sf*Jcint|
        | -Sm*T1 kp*Sm*Jcit  0  0 |     |-kp*Sd*Jcint|
        |    T0      0       0  0 |     |     0      |
        |    I       0       0  0 |     |     0      |
        |    0       0       0  0 |     |     0      |

    xa = | dxi q xi -g qra|   u = dqr

    Aa = | Ad  Bd|  Ba = | Bd|
         | 0   I |       | I |

    Ca = |0 I 0 0 0| y = |q|
    */

    auto roty = RobotMtx->Rot_mtx;

    auto Jc = roty * RobotMtx->J_toe;
    RobotMtx->CoMJacobian();
    RobotMtx->CoMPos();

    auto Jcom = roty * RobotMtx->J_com;
    auto r_ = roty * RobotMtx->CoM;

    auto pc = roty * RobotMtx->HT_toe.block(0, 3, 3, 1);

    auto r_pc = pc - r_; // ok

    Eigen::Matrix3d Jcs;
    Jcs << Jc(0, 0), Jc(0, 1), Jc(0, 2), Jc(2, 0), Jc(2, 1), Jc(2, 2), 1, 1, 1; // ok

    auto Jc_invT = (Jcs.transpose()).inverse();

    auto gamma = Jcom - Jc;

    Eigen::Matrix3d alpha;
    alpha << 1, 0, r_pc(2, 0),
        0, 1, -r_pc(0, 0),
        0, 0, 1;

    Eigen::Matrix3d beta;
    beta << gamma(0, 0), gamma(0, 1), gamma(0, 2), gamma(2, 0), gamma(2, 1), gamma(2, 2), 1, 1, 1;

    auto inertia = roty * RobotMtx->InertiaTensor();

    sum_m << r_pc(2, 0), -r_pc(0, 0), -1;
    sum_m = sum_m / inertia(1, 1);

    auto T0 = beta.inverse() * alpha;
    auto T1 = Kd * Jc_invT * T0;

    A.block(0, 0, 2, 3) = -sum_f * T1;
    A.block(0, 3, 2, 3) = Kp * sum_f * Jc_invT;

    A.block(2, 0, 1, 3) = -sum_m * T1;
    A.block(2, 3, 1, 3) = Kp * sum_m * Jc_invT;

    A.block(3, 0, 3, 3) = T0;

    B.block(0, 0, 2, 3) = -Kp * sum_f * Jc_invT;
    B.block(2, 0, 1, 3) = -Kp * sum_m * Jc_invT;

    /*
     Using forward Euler, the dynamical model is dicretized and then aumented
      Aa = | Ad  Bd|  Ba = | Bd|
           | 0   I |       | 0 |
    */

    Aa.block(0, 0, 10, 10) = Eigen::MatrixXd::Identity(10, 10) + ts * A;
    Aa.block(0, 10, 10, 3) = ts * B;
    Ba.block(0, 0, 10, 3) = ts * B;

    // Update the constraint model
    this->C_cons.block(3, 0, 3, 3) = -Kd * T0;
}

void OptProblem5::DefineConstraintMtxs()
{

    this->Phi_cons.block(0, 0, 4, this->nxa) = this->C_cons.block(0, 0, 4, this->nxa) * this->Aa;
    this->Phi_cons.block(3, 0, 3, this->nxa) = this->C_cons.block(3, 0, 3, this->nxa);
    this->aux_cons.block(0, 0, 4, this->nu) = this->C_cons.block(0, 0, 4, this->nxa) * this->Ba;
    this->aux_cons.block(3, 0, 3, this->nu) = Kp * Eigen::MatrixXd::Identity(3, 3);

    // std::cout << this->C_cons << std::endl;
    // std::cout << this->Phi_cons.block(0, 0, 7, this->nxa) << std::endl;

    // auto roty = RobotMtx->Rot_mtx;

    // auto toe_pos_x = roty * (RobotMtx->HT_toe).block(0, 3, 3, 1);
    // auto heel_pos_x = roty * (RobotMtx->HT_heel).block(0, 3, 3, 1);

    // this->Ucv_var = RobotMtx->b.block(0, 0, 1, 1) + toe_pos_x.block(0, 0, 1, 1);
    // this->Lcv_var = RobotMtx->b.block(0, 0, 1, 1) + heel_pos_x.block(0, 0, 1, 1);

    // this->ref(0, 0) = 0.25;
    // this->UpdateReferences(this->ref);
}