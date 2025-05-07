#ifndef OPT_PROBLEM1_H
#define OPT_PROBLEM1_H

#include "rgc_controller/opt_problem.h"

class OptProblem1 : public OptProblem
{
public:
    OptProblem1(ModelMatrices *Robot);

    ~OptProblem1();

    void UpdateDynamicModel() override;

    void UpdateModelConstants() override;

    void DefineConstraintMtxs() override;

    ModelMatrices *RobotMtx;

    Eigen::MatrixXd C_cons_aux, GRF_mtx, sum_f, sum_m;

    Eigen::Matrix<double, 3, 1> n1;
    Eigen::Matrix<double, 3, 1> t1;
};

#endif