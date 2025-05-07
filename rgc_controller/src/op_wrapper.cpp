#include "rgc_controller/op_wrapper.h"

Op_Wrapper::Op_Wrapper()
{

    this->solver.settings()->setVerbosity(false);

    this->_JumpRobot = new ModelMatrices();

    this->optP0 = new OptProblem0(_JumpRobot);
    this->op[0] = this->optP0;

    this->optP1 = new OptProblem1(_JumpRobot);
    this->op[1] = this->optP1;

    this->optP2 = new OptProblem2(_JumpRobot);
    this->op[2] = this->optP2;

    // this->optP3 = new OptProblem3(_JumpRobot);
    // this->op[3] = this->optP3;

    // this->optP4 = new OptProblem4(_JumpRobot);
    // this->op[4] = this->optP4;

    // this->optP5 = new OptProblem5(_JumpRobot);
    // this->op[5] = this->optP5;

    // this->optP6 = new OptProblem6(_JumpRobot);
    // this->op[6] = this->optP6;

    // this->optP7 = new OptProblem7(_JumpRobot);
    // this->op[7] = this->optP7;

    this->qhl.resize(3, 1);
}

Op_Wrapper::~Op_Wrapper()
{
}

void Op_Wrapper::RGCConfig(double _ts, double _Kp, double _Kd)
{
    std::cout << "Configuring POs" << std::endl;

    // std::vector<int> indices = {0, 1, 2, 3, 4, 5, 6, 7}; // All relevant indices
    std::vector<int> indices = {0, 1, 2}; // All relevant indices

    for (int i : indices)
    {
        this->op[i]->SetInternalVariables();
        this->op[i]->SetConstants(_ts, N_matrices[i], M_matrices[i], _Kp, _Kd);
        this->op[i]->UpdateModelConstants();

        this->ConfPO(i);

        // Directly use vector indexing (faster than unordered_map lookup)
        this->op[i]->SetWeightMatrices(this->Q_matrices[i], this->R_matrices[i]);

        this->ClearPO();
        this->ClearData();
    }

    this->first_conf = 1;
}

void Op_Wrapper::LoadConfig(const std::string &filename)
{
    YAML::Node config = YAML::LoadFile(filename);

    // Resize vectors to store 8 elements (indices 0 to 7)
    Q_matrices.resize(9);
    R_matrices.resize(9);
    N_matrices.resize(9);
    M_matrices.resize(9);

    // Load Q and R diagonal values and construct matrices
    for (int i = 0; i < 9; i++)
    {
        std::string key = std::to_string(i);
        if (config["PO"][key])
        {
            YAML::Node op_node = config["PO"][key];

            // Load Q diagonal elements
            std::vector<double> Q_diag = op_node["Q"].as<std::vector<double>>();
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(Q_diag.size(), Q_diag.size());
            for (size_t j = 0; j < Q_diag.size(); j++)
            {
                Q(j, j) = Q_diag[j]; // Set diagonal elements
            }
            Q_matrices[i] = Q;

            // Load R diagonal elements
            std::vector<double> R_diag = op_node["R"].as<std::vector<double>>();
            Eigen::MatrixXd R = Eigen::MatrixXd::Zero(R_diag.size(), R_diag.size());
            for (size_t j = 0; j < R_diag.size(); j++)
            {
                R(j, j) = R_diag[j]; // Set diagonal elements
            }
            R_matrices[i] = R;
            N_matrices[i] = op_node["N"].as<int>();
            M_matrices[i] = op_node["M"].as<int>();
        }
    }

    std::cout << "RGC Configuration Loaded Successfully from " << filename << std::endl;
}

void Op_Wrapper::UpdateSt(Eigen::Matrix<double, 3, 1> *_q,
                          Eigen::Matrix<double, 3, 1> *_qd,
                          Eigen::Matrix<double, 3, 1> *_qr,
                          Eigen::Matrix<double, 2, 1> *_dr,
                          Eigen::Matrix<double, 2, 1> *_r,
                          Eigen::Matrix<double, 2, 1> *_db,
                          Eigen::Matrix<double, 2, 1> *_b,
                          double _dth,
                          double _th)
{
    q = (*_q);
    qd = (*_qd);
    qr = (*_qr);
    r_vel = (*_dr);
    r_pos = (*_r);
    dth = _dth;
    th = _th;
    _JumpRobot->UpdateRobotStates(*_q, *_qd, th, *_b, *_db);
}

int Op_Wrapper::ChooseRGCPO(int npo)
{
    if (npo != this->last_op)
    {
        if (this->solver.isInitialized())
            this->ClearPO();

        if (this->solver.data()->isSet())
            this->ClearData();

        this->ConfPO(npo);
        this->last_op = npo;
        error_flag = false;
    }

    if (this->solver.isInitialized())
    {
        if (npo == 0 || npo == 1 || npo == 2)
        {
            // update the states vector |dr, dth, q, g, qa|
            this->x << r_vel, dth, q, r_pos, th, g, qr;
        }

        // if (npo == 0 || npo == 1 || npo == 2 || npo == 3 || npo == 4 || npo == 5)
        // {
        //     // update the states vector |dr, dth, q, g, qa|
        //     this->x << r_vel, dth, q, r_pos, th, g, qr;
        // }

        // if (npo == 6 || npo == 7)
        // {
        //     // update the states vector |dq q dth th r qra|
        //     this->x << this->qd, this->q, dth, th, r_pos, this->qr;
        // }

        // if (npo == 4)
        // {

        //     auto psi = this->q(0, 0) + this->q(1, 0) + this->q(2, 0) + th / 2;
        //     // update the states vector |dq q dth th r qra|
        //     this->x << this->qd, this->q, dth, th, r_pos, psi, this->qr;
        // }

        // this->qhl = this->op[npo]->qhl;
        this->op[npo]->UpdateStates(this->x);

        this->op[npo]->UpdateOptimizationProblem(this->H, this->F, this->Ain, this->Lb, this->Ub, this->G_st, this->Phi_st);

        int solve_status = this->SolvePO();
        if (solve_status == 1)
        {
            if (this->debug)
                std::cout << "solved" << std::endl;
            return 1;
        }
        else if (solve_status == 0)
        {
            if (this->debug)
                std::cout << "not solved" << std::endl;
            return 0;
        }
        else
        {
            return -1;
        }
    }
    else
    {
        if (this->debug)
            std::cout << "RGC conf error" << std::endl;
        return -1;
    }
}

void Op_Wrapper::ResetPO()
{
    if (this->solver.isInitialized())
        this->ClearPO();
    if (this->solver.data()->isSet())
        this->ClearData();
    this->last_op = -1;
}

void Op_Wrapper::ClearPO()
{
    this->solver.clearSolverVariables();
    this->solver.clearSolver();
}

void Op_Wrapper::ClearData()
{
    this->solver.data()->clearLinearConstraintsMatrix();
    this->solver.data()->clearHessianMatrix();
}

void Op_Wrapper::ConfPO(int index)
{
    // first, resize the matrices

    this->x.resize(this->op[index]->nxa);
    this->x.setZero();

    this->H.resize(this->op[index]->nu * this->op[index]->M, this->op[index]->nu * this->op[index]->M);
    this->H.setZero();

    this->F.resize(1, this->op[index]->nu * this->op[index]->M);
    this->F.setZero();

    // std::cout << "Numer of constraints: " << this->op[index]->nc << std::endl;

    this->Ain.resize(this->op[index]->nc * this->op[index]->N, this->op[index]->nu * this->op[index]->M);
    this->Ain.setZero();

    this->Lb.resize(this->op[index]->nc * this->op[index]->N);
    this->Lb.setZero();

    this->Ub.resize(this->op[index]->nc * this->op[index]->N);
    this->Ub.setZero();

    this->Phi_st.resize(this->op[index]->nst * this->op[index]->N, this->op[index]->nxa);
    this->Phi_st.setZero();

    this->G_st.resize(this->op[index]->nst * this->op[index]->N, this->op[index]->nu * this->op[index]->M);
    this->G_st.setZero();

    this->x_pred.resize(this->op[index]->nst * this->op[index]->N, 1);
    this->x_pred.setZero();

    // then, configure the solver

    this->solver.settings()
        ->setVerbosity(0);

    this->solver.data()->setNumberOfVariables(this->op[index]->nu * this->op[index]->M);

    this->hessian_sparse = this->H.sparseView();
    this->solver.data()->clearHessianMatrix();
    this->solver.data()->setHessianMatrix(this->hessian_sparse);

    this->solver.data()->setGradient(F.transpose());

    this->solver.data()->setNumberOfConstraints(this->op[index]->nc * this->op[index]->N);
    this->linearMatrix = this->Ain.sparseView();
    this->solver.data()->setLinearConstraintsMatrix(this->linearMatrix);
    this->solver.data()->setLowerBound(this->Lb);
    this->solver.data()->setUpperBound(this->Ub);

    if (this->op[index]->nc != 0)
        this->constraints = 1;

    if (!this->first_conf)
    {
        if (!this->solver.initSolver())
            std::cout << "***************** PO " << index << " Inicialization Problem ***************** " << std::endl;
        else
            std::cout << "***************** PO " << index << " OK ***************** " << std::endl;
    }
    else
    {
        if (!this->solver.initSolver())
        {
            std::cout << "Error: " << index << std::endl;
        }
    }
}

int Op_Wrapper::SolvePO()
{

    this->hessian_sparse = this->H.sparseView();
    if (!this->solver.updateHessianMatrix(this->hessian_sparse))
        return -1;

    this->solver.updateGradient(this->F.transpose());

    if (this->constraints != 0)
    {
        this->linearMatrix = this->Ain.sparseView();
        this->solver.updateLinearConstraintsMatrix(this->linearMatrix);
        this->solver.updateBounds(this->Lb, this->Ub);
    }

    if (this->solver.solveProblem() == OsqpEigen::ErrorExitFlag::NoError)
    {
        if (this->solver.getStatus() != OsqpEigen::Status::Solved)
        {
            this->x_pred = this->Phi_st * this->x;
            return 0;
        }

        this->QPSolution = this->solver.getSolution();
        this->delta_qr = this->QPSolution.block(0, 0, 3, 1);
        obj_val = this->solver.getObjValue();
        this->x_pred = this->G_st * this->QPSolution + this->Phi_st * this->x;
        // std::cout << obj_val << std::endl;
        return 1;
    }
    else
    {
        if (this->debug)
            std::cout << "Not solved - error" << std::endl;
        return 0;
    }
}