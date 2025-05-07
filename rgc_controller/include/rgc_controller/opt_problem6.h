#ifndef OPT_PROBLEM6_H
#define OPT_PROBLEM6_H

#include "rgc_controller/opt_problem.h"

class OptProblem6 : public OptProblem
{
public:
    OptProblem6(ModelMatrices *Robot);

    ~OptProblem6();

    void UpdateDynamicModel() override;

    void UpdateModelConstants() override;

    void DefineConstraintMtxs() override;

    ModelMatrices *RobotMtx;

    Eigen::MatrixXd C_cons_aux, GRF_mtx;

    Eigen::Matrix<double, 2, 1> n1;
    Eigen::Matrix<double, 2, 1> t1;
};

#endif