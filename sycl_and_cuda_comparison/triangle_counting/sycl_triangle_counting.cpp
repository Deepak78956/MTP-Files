#include <sycl/sycl.hpp>
#include <array>
#include <iostream>
#include <string>
using namespace std;

int main()
{
    sycl::queue Q{sycl::gpu_selector{}};

    sycl::device dev = Q.get_device();
    cout << "Device Name: " << dev.get_info<sycl::info::device::name>() << endl;
    cout << "Device Vendor: " << dev.get_info<sycl::info::device::vendor>() << endl;
}