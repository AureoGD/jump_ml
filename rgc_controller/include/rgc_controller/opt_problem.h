#ifndef OPT_PROBLEM_H
#define OPT_PROBLEM_H

#include "rgc_controller/pred_control.h"
#include "rgc_controller/model_matrices.h"

class OptProblem : public PredControl
{
public:
    virtual void SetConstants(double _ts, int _N, int _M, double _kp, double _kd);

    virtual void UpdateModelConstants() = 0;

    virtual void UpdateReferences(Eigen::VectorXd ref);

    Eigen::VectorXd ref;

    double ts, Kp, Kd;
};

#endif