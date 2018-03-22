//
// Created by chnlkw on 1/18/18.
//

//#include "Context.h"
//#include "DataCopy.h"
//
//Context::Context() :
//        device_(-1),
//        allocator_([](int device) { return AllocatorPtr(new CudaAllocator(device)); }) {
//    CUDA_CALL(cudaGetDeviceCount, &num_devices_);
//    DataCopyInitP2P();
//}
//
//int Context::Device() const {
//    return device_;
//}
//
//int Context::GetNumDevices() const {
//    return num_devices_;
//}
//
//void Context::SetDevice(int device_id) {
//    device_ = device_id;
//}
//
//void Context::SetStream(cudaStream_t stream) {
//    stream_ = stream;
//}
//
//cudaStream_t Context::GetStream() const {
//    return stream_;
//}
//
//AllocatorPtr Context::GetAllocator() {
//    return allocator_.GetAllocatorByDevice(device_);
//}

//void *Context::Alloc(size_t size) {
//    return allocator_.Alloc(size, device_);
//}
//
//void Context::Free(void *ptr) {
//    allocator_.Free(ptr);
//}

//Context g_context;

//DevicePtr g_device(new CPUDevice);

//DevicePtr Device::cpu(new CPUDevice);
//DevicePtr Device::current = Device::cpu;