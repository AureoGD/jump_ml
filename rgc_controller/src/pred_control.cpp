#include "rgc_controller/pred_control.h"

void PredControl::SetInternalVariables()
{
    this->nx = this->A.rows();        // # states
    this->nu = this->B.cols();        // # decision variables
    this->ny = this->Ca.rows();       // # outputs for the optimization problem
    this->nxa = this->Aa.rows();      // # aumented states
    this->nc = this->C_cons.rows();   // # constraints
    this->ncv = this->C_consV.cols(); // # variable constraints
    this->nst = this->C_st.rows();    // # variable constraints

    // std::cout << this->nc << std::endl;
}

void PredControl::ResizeMatrices()
{
    // Model prediction matrices: y̅ = G*u̅ + Phi*x[k]
    this->x.resize(this->nxa, 1);
    this->x.setZero();

    this->G.resize(this->ny * this->N, this->nu * this->M);
    this->G.setZero();

    this->Phi.resize(this->ny * N, this->nxa);
    this->Phi.setZero();

    // Auxiliare matrices
    this->aux_mdl.resize(this->ny, this->nu);
    this->aux_mdl.setZero();

    this->aux_cons.resize(this->nc, this->nu);
    this->aux_cons.setZero();

    this->aux_st.resize(this->nst, this->nu);
    this->aux_st.setZero();

    // Constraist prediction mode:
    // Lcv_block*Lcv_var + Lc_block - Phi_cons*x[k] <= G_cons*u̅ <= Uc_block + Ucv_block *Ucv_var- Phi_cons*x[k]
    this->G_cons.resize(this->nc * this->N, this->nu * this->M);
    this->G_cons.setZero();

    this->Phi_cons.resize(this->nc * N, this->nxa);
    this->Phi_cons.setZero();

    this->Uc_block.resize(this->nc * this->N, 1);
    this->Uc_block.setZero();

    this->Lc_block.resize(this->nc * this->N, 1);
    this->Lc_block.setZero();

    this->Lcv_block.resize(this->nc * this->N, this->ncv);
    this->Ucv_block.resize(this->nc * this->N, this->ncv);

    this->Lcv_var.resize(this->ncv, 1);
    this->Ucv_var.resize(this->ncv, 1);

    // Create a predict model of some states
    this->G_st.resize(this->nst * this->N, this->nu * this->M);
    this->G_st.setZero();

    this->Phi_st.resize(this->nst * N, this->nxa);
    this->Phi_st.setZero();

    // Weight matrices: The cost function must be writen as J = (r̅-y̅)'*Q_blcok*(r̅-y̅) + u̅'*R_block*u̅

    this->Ref_block.resize(this->ny * this->N, 1);
    this->Ref_block.setZero();

    this->Q_block.resize(this->ny * this->N, this->ny * this->N);
    this->Q_block.setZero();

    this->R_block.resize(this->nu * this->M, this->nu * this->M);
    this->R_block.setZero();
}

void PredControl::SetWeightMatrices(const Eigen::MatrixXd &Q, const Eigen::MatrixXd &R)
{
    if (Q.rows() != this->ny || Q.cols() != this->ny)
    {
        std::cout << "Wrong Q dimension!" << std::endl;
        std::cout << "Expected: " << this->ny << "x" << this->ny << " - Receive: " << Q.rows() << "x" << Q.cols() << std::endl;

        return;
    }

    if (R.rows() != this->nu || R.cols() != this->nu)
    {
        std::cout << "Wrong R dimension!" << std::endl;
        std::cout << "Expected: " << this->nu << "x" << this->nu << " - Receive: " << R.rows() << "x" << R.cols() << std::endl;
        return;
    }

    for (int i = 0; i < this->N; i++)
    {
        this->Q_block.block(i * this->ny, i * this->ny, this->ny, this->ny) = Q;
        if (i < this->M)
            this->R_block.block(i * this->nu, i * this->nu, this->nu, this->nu) = R;
    }
}

void PredControl::SetConsBounds(const Eigen::MatrixXd &Lb, const Eigen::MatrixXd &Ub)
{
    if (Lb.rows() != this->nc || Lb.cols() != 1)
    {
        std::cout << "Wrong Lower Bound dimension" << std::endl;
        std::cout << "Expected: " << this->nc << "x1" << " - Receive: " << Lb.rows() << "x" << Lb.cols() << std::endl;

        return;
    }

    if (Ub.rows() != this->nc || Ub.cols() != 1)
    {
        std::cout << "Wrong Upper Bound dimension" << std::endl;
        std::cout << "Expected: " << this->nc << "x1" << " - Receive: " << Ub.rows() << "x" << Ub.cols() << std::endl;
        return;
    }

    this->Lc = Lb;
    this->Uc = Ub;

    for (int i = 0; i < this->N; i++)
    {
        this->Lc_block.block(i * this->nc, 0, this->nc, 1) = this->Lc;
        this->Uc_block.block(i * this->nc, 0, this->nc, 1) = this->Uc;
        this->Lcv_block.block(i * this->nc, 0, this->nc, this->ncv) = this->C_consV;
        this->Ucv_block.block(i * this->nc, 0, this->nc, this->ncv) = this->C_consV;
    }
}

void PredControl::SetReference(const Eigen::MatrixXd &ref)
{
    if (ref.rows() != this->ny || ref.cols() != 1)
    {
        std::cout << "Wrong Ref dimension!" << std::endl;
        std::cout << "Expected: " << this->ny << "x1" << " - Receive: " << ref.rows() << "x" << ref.cols() << std::endl;
        return;
    }

    for (int i = 0; i < this->N; i++)
    {
        this->Ref_block.block(i * this->ny, 0, this->ny, 1) = ref;
    }
}

void PredControl::UpdateStates(const Eigen::VectorXd &x)
{
    this->x = x;
}

void PredControl::UpdateOptimizationProblem(Eigen::MatrixXd &H,
                                            Eigen::MatrixXd &F,
                                            Eigen::MatrixXd &Ain,
                                            Eigen::VectorXd &lowerBound,
                                            Eigen::VectorXd &upperBound,
                                            Eigen::MatrixXd &_G_st,
                                            Eigen::MatrixXd &_Phi_st)
{
    // Update the dynamic matrices
    this->UpdateDynamicModel();

    // define initial Phi values
    this->DefinePhi();

    // update the prediction model
    this->UpdatePredictionModel();

    // Update the optimization problem

    H = 2 * (this->G.transpose() * this->Q_block * this->G + this->R_block);

    F = 2 * (((this->Phi * this->x) - this->Ref_block).transpose()) * this->Q_block * this->G;
    Ain = this->G_cons;

    // lowerBound = this->Lc_block - this->Phi_cons * this->x;
    // upperBound = this->Uc_block - this->Phi_cons * this->x;

    lowerBound = this->Lc_block - this->Phi_cons * this->x + this->Lcv_block * this->Lcv_var;
    upperBound = this->Uc_block - this->Phi_cons * this->x + this->Ucv_block * this->Ucv_var;

    _G_st = this->G_st;
    _Phi_st = this->Phi_st;
}

void PredControl::DefinePhi()
{
    this->Phi.block(0, 0, this->ny, this->nxa) = this->Ca * this->Aa;
    this->Phi_st.block(0, 0, this->nst, this->nxa) = this->C_st * this->Aa;
    this->aux_mdl = this->Ca * this->Ba;
    this->aux_st = this->C_st * this->Ba + this->D_st;
}

void PredControl::DefineConstraintMtxs()
{
}

void PredControl::UpdatePredictionModel()
{
    if (this->nc == 0)
    {
        for (int i = 0; i < N; i++)
        {
            int j = 0;

            if (i != 0)
            {
                this->Phi.block(i * this->ny, 0, this->ny, this->nxa) = this->Phi.block((i - 1) * this->ny, 0, this->ny, this->nxa) * this->Aa;
                this->aux_mdl = this->Phi.block((i - 1) * this->ny, 0, this->ny, this->nxa) * this->Ba;
            }

            while ((j < this->M) and (i + j < this->N))
            {
                this->G.block((i + j) * this->ny, j * this->nu, this->ny, this->nu) = this->aux_mdl;
                j++;
            }
        }
    }
    else
    {
        this->DefineConstraintMtxs();
        // this->Phi_cons.block(0, 0, this->nc, this->nxa) = this->C_cons * this->Aa;
        // this->aux_cons = this->C_cons * this->Ba;
        for (int i = 0; i < N; i++)
        {
            int j = 0;

            if (i != 0)
            {
                this->Phi.block(i * this->ny, 0, this->ny, this->nxa) = this->Phi.block((i - 1) * this->ny, 0, this->ny, this->nxa) * this->Aa;
                this->aux_mdl = this->Phi.block((i - 1) * this->ny, 0, this->ny, this->nxa) * this->Ba;

                this->Phi_cons.block(i * this->nc, 0, this->nc, this->nxa) = this->Phi_cons.block((i - 1) * this->nc, 0, this->nc, this->nxa) * this->Aa;
                this->aux_cons = this->Phi_cons.block((i - 1) * this->nc, 0, this->nc, this->nxa) * this->Ba;

                this->Phi_st.block(i * this->nst, 0, this->nst, this->nxa) = this->Phi_st.block((i - 1) * this->nst, 0, this->nst, this->nxa) * this->Aa;
                this->aux_st = this->Phi_st.block((i - 1) * this->nst, 0, this->nst, this->nxa) * this->Ba;
            }

            while ((j < this->M) and (i + j < this->N))
            {
                this->G.block((i + j) * this->ny, j * this->nu, this->ny, this->nu) = this->aux_mdl;
                this->G_cons.block((i + j) * this->nc, j * this->nu, this->nc, this->nu) = this->aux_cons;
                this->G_st.block((i + j) * this->nst, j * this->nu, this->nst, this->nu) = this->aux_st;
                j++;
            }
        }
    }
}