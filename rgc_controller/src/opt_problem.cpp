#include "rgc_controller/opt_problem.h"

void OptProblem::SetConstants(double _ts, int _N, int _M, double _kp, double _kd)
{
    this->ts = _ts;
    this->Kp = _kp;
    this->Kd = _kd;

    this->N = _N;
    this->M = _M;

    this->ResizeMatrices();
}

void OptProblem::UpdateModelConstants()
{
}

void OptProblem::UpdateReferences(Eigen::VectorXd ref)
{
    this->SetReference(ref);
}
