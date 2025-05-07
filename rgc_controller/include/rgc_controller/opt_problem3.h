#ifndef OPT_PROBLEM3_H
#define OPT_PROBLEM3_H

#include "rgc_controller/opt_problem.h"

class OptProblem3 : public OptProblem
{
public:
    OptProblem3(ModelMatrices *Robot);

    ~OptProblem3();

    void UpdateDynamicModel() override;

    void UpdateModelConstants() override;

    void DefineConstraintMtxs() override;

    ModelMatrices *RobotMtx;

    Eigen::MatrixXd C_cons_aux, GRF_mtx, sum_f, sum_m;

    Eigen::Matrix<double, 3, 1> n1;
    Eigen::Matrix<double, 3, 1> t1;
};

#endif