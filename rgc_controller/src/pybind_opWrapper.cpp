#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "rgc_controller/op_wrapper.h"
#include "rgc_controller/model_matrices.h"

namespace py = pybind11;

PYBIND11_MODULE(pybind_opWrapper, m)
{
    py::class_<Op_Wrapper>(m, "Op_Wrapper")
        .def(py::init<>())
        .def("RGCConfig", &Op_Wrapper::RGCConfig)
        .def("load_config", &Op_Wrapper::LoadConfig)
        .def("UpdateSt", &Op_Wrapper::UpdateSt)
        .def("ChooseRGCPO", &Op_Wrapper::ChooseRGCPO)
        .def("ResetPO", &Op_Wrapper::ResetPO)
        // .def_readwrite("com_pos_w", &Op_Wrapper::r_pos)
        // .def_readwrite("com_vel", &Op_Wrapper::r_vel)
        // .def_readwrite("foot_pos", &Op_Wrapper::foot_pos)
        // .def_readwrite("foot_vel", &Op_Wrapper::foot_vel)
        .def_readwrite("delta_qr", &Op_Wrapper::delta_qr)
        .def_readwrite("delta_qhl", &Op_Wrapper::qhl)
        .def_readwrite("obj_val", &Op_Wrapper::obj_val)
        .def_readwrite("pred_states", &Op_Wrapper::x_pred);
}
