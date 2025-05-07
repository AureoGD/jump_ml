#ifndef PRED_CONTROL_H
#define PRED_CONTROL_H

#include <OsqpEigen/OsqpEigen.h>
#include "rgc_controller/pred_control.h"
#include "math.h"

class PredControl
{
public:
    virtual void UpdateOptimizationProblem(Eigen::MatrixXd &H,
                                           Eigen::MatrixXd &F,
                                           Eigen::MatrixXd &Ain,
                                           Eigen::VectorXd &lowerBound,
                                           Eigen::VectorXd &upperBound,
                                           Eigen::MatrixXd &_G_st,
                                           Eigen::MatrixXd &_Phi_st);

    virtual void UpdateDynamicModel() = 0;

    virtual void UpdatePredictionModel();

    virtual void UpdateStates(const Eigen::VectorXd &x);

    virtual void SetInternalVariables();

    virtual void ResizeMatrices();

    virtual void SetWeightMatrices(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R);

    virtual void SetConsBounds(const Eigen::MatrixXd &Lb, const Eigen::MatrixXd &Ub);

    virtual void SetReference(const Eigen::MatrixXd &ref);

    virtual void DefinePhi();

    virtual void DefineConstraintMtxs();

    Eigen::MatrixXd A, B, Aa, Ba, Ca, C_cons, C_consV, C_st, D_st;

    Eigen::VectorXd Uc, Lc;

    int N, M;

    const double PI = std::atan(1.0) * 4;

    // private:
    Eigen::MatrixXd G, Phi, G_cons, Phi_cons, Phi_st, G_st, Q_block, R_block, Uc_block, Lc_block, Ref_block, aux_mdl, aux_cons, aux_st, Ucv_block, Lcv_block, Lcv_var, Ucv_var;

    Eigen::VectorXd x;

    int nx, nxa, ny, nu, nc, ncv, nst;
};
#endif