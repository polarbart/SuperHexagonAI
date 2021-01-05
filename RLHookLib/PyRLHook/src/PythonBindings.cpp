// https://docs.microsoft.com/de-de/visualstudio/python/working-with-c-cpp-python-in-visual-studio?view=vs-2019

#define NOMINMAX 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

#include "GameInterface.h"
#include "Utils.h"

namespace py = pybind11;

PYBIND11_MODULE(pyrlhook, m) {
	py::class_<GameInterface>(m, "GameInterface")
		.def(py::init<const std::string&, PixelFormat, PixelDataType>())
		.def(py::init<DWORD, PixelFormat, PixelDataType>())
		.def("step", &GameInterface::step)
		.def_property_readonly("buffer_size", &GameInterface::bufferSize)
		.def("set_speed", &GameInterface::setSpeed)
		.def("run_afap", &GameInterface::runAfap)
        .def("read_byte", py::overload_cast<const std::string&, const std::vector<DWORD>&>(&GameInterface::read<BYTE>))
        .def("read_byte", py::overload_cast<const LPCVOID>(&GameInterface::read<BYTE>))
        .def("read_dword", py::overload_cast<const std::string&, const std::vector<DWORD>&>(&GameInterface::read<DWORD>))
		.def("read_dword", py::overload_cast<const LPCVOID>(&GameInterface::read<LPCVOID>))
		.def("read_double", py::overload_cast<const std::string&, const std::vector<DWORD>&>(&GameInterface::read<std::double_t>))
		.def("read_double", py::overload_cast<const LPCVOID>(&GameInterface::read<std::double_t>))
		.def("write_byte", py::overload_cast<const std::string&, const std::vector<DWORD>&, const BYTE&>(&GameInterface::write<BYTE>))
		.def("write_byte", py::overload_cast<const LPVOID, const BYTE&>(&GameInterface::write<BYTE>))
		.def("write_dword", py::overload_cast<const std::string&, const std::vector<DWORD>&, const DWORD&>(&GameInterface::write<DWORD>))
		.def("write_dword", py::overload_cast<const LPVOID, const DWORD&>(&GameInterface::write<DWORD>))
        .def("write_double", py::overload_cast<const std::string&, const std::vector<DWORD>&, const std::double_t&>(&GameInterface::write<std::double_t>))
        .def("write_double", py::overload_cast<const LPVOID, const std::double_t&>(&GameInterface::write<std::double_t>));

	py::enum_<PixelFormat>(m, "PixelFormat")
            .value("RED", PixelFormat::RED)
            .value("GREEN", PixelFormat::GREEN)
            .value("BLUE", PixelFormat::BLUE)
            .value("ALPHA", PixelFormat::ALPHA)
            .value("RGB", PixelFormat::RGB)
            .value("RGBA", PixelFormat::RGBA);

	py::enum_<PixelDataType>(m, "PixelDataType")
	        .value("UINT8", PixelDataType::UBYTE)
	        .value("FLOAT32", PixelDataType::FLOAT32);

    // py::scoped_interpreter guard {};
    py::dict locals;
    auto path = py::module::import("sys").attr("prefix").cast<std::string>().append("\\Lib\\site-packages");
	GameInterface::basePath = path;
}