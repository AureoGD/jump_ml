#ifndef OPT_PROBLEM7_H
#define OPT_PROBLEM7_H

#include "rgc_controller/opt_problem.h"

class OptProblem7 : public OptProblem
{
public:
    OptProblem7(ModelMatrices *Robot);

    ~OptProblem7();

    void UpdateDynamicModel() override;

    void UpdateModelConstants() override;

    void DefineConstraintMtxs() override;

    ModelMatrices *RobotMtx;

    Eigen::MatrixXd C_cons_aux, GRF_mtx;

    Eigen::Matrix<double, 2, 1> n1;
    Eigen::Matrix<double, 2, 1> t1;
};

#endif