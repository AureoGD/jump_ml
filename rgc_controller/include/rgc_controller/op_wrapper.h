#ifndef OP_WRAPPER_H
#define OP_WRAPPER_H

#include "rgc_controller/pred_control.h"
#include "rgc_controller/opt_problem.h"
#include "rgc_controller/opt_problem0.h"
#include "rgc_controller/opt_problem1.h"
#include "rgc_controller/opt_problem2.h"
// #include "rgc_controller/opt_problem3.h"
// #include "rgc_controller/opt_problem4.h"
// #include "rgc_controller/opt_problem5.h"
// #include "rgc_controller/opt_problem6.h"
// #include "rgc_controller/opt_problem7.h"

#include "rgc_controller/model_matrices.h"

#include "yaml-cpp/yaml.h"
#include <filesystem> // To handle paths safely

#include <OsqpEigen/OsqpEigen.h>
#include "math.h"

class Op_Wrapper
{
public:
    Op_Wrapper();

    ~Op_Wrapper();

    void RGCConfig(double _ts, double _Kp, double _Kd);

    int ChooseRGCPO(int npo);

    int SolvePO();

    void UpdateSt(Eigen::Matrix<double, 3, 1> *_q,
                  Eigen::Matrix<double, 3, 1> *_qd,
                  Eigen::Matrix<double, 3, 1> *_qr,
                  Eigen::Matrix<double, 2, 1> *_dr,
                  Eigen::Matrix<double, 2, 1> *_r,
                  Eigen::Matrix<double, 2, 1> *_db,
                  Eigen::Matrix<double, 2, 1> *_b,
                  double dth,
                  double _th);

    void ResetPO();

    void ClearData();

    double dth, th;

    Eigen::VectorXd qhl;
    Eigen::VectorXd x_pred;

    Eigen::Matrix<double, 3, 1> q, qd, qr, delta_qr;
    Eigen::VectorXd QPSolution;

    Eigen::Matrix<double, 2, 1> r_vel;
    Eigen::Matrix<double, 2, 1> r_pos;
    Eigen::Matrix<double, 2, 1> foot_pos;
    Eigen::Matrix<double, 2, 1> foot_vel;

    void LoadConfig(const std::string &filename);
    double obj_val;

private:
    void ConfPO(int index);

    void ClearPO();

    OptProblem0 *optP0;
    OptProblem1 *optP1;
    OptProblem2 *optP2;
    // OptProblem3 *optP3;
    // OptProblem4 *optP4;
    // OptProblem5 *optP5;
    // OptProblem6 *optP6;
    // OptProblem7 *optP7;

    ModelMatrices *_JumpRobot;

    OptProblem *op[8];

    int last_op = -1;

    const double g = -9.81;

    bool constraints = 0;

    bool first_conf = 0;

    Eigen::MatrixXd H, F, Ain;

    Eigen::MatrixXd G_st, Phi_st;

    Eigen::VectorXd x, Ub, Lb;

    Eigen::SparseMatrix<double> hessian_sparse, linearMatrix;

    OsqpEigen::Solver solver;

    bool debug = false;

    bool error_flag = false;

    std::vector<Eigen::MatrixXd> Q_matrices;
    std::vector<Eigen::MatrixXd> R_matrices;
    std::vector<int> N_matrices;
    std::vector<int> M_matrices;
};

#endif