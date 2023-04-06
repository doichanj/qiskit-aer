/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _clifford_thrust_hpp_
#define _clifford_thrust_hpp_

#include "framework/types.hpp"
#include "framework/utils.hpp"

#include "clifford.hpp"
#include "binary_matrix_thrust.hpp"
#include "framework/json_parser.hpp"

#include "simulators/statevector/chunk/thrust_kernels.hpp"

#include <omp.h>

namespace AER {
namespace Clifford {

//operator type
struct CliffordOp {
  char op;
  uint_t tar;
  uint_t con;
};

#define CLIFFORD_OPS_BUFFER_SIZE  2048
#define CLIFFORD_PARAMS_BUFFER_SIZE 32

/*******************************************************************************
 *
 * Clifford Class for Thrust
 *
 ******************************************************************************/

class CliffordThrust {
public:
  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------
  CliffordThrust() = default;
  CliffordThrust(CliffordThrust& src);

  ~CliffordThrust();

  CliffordThrust& operator=(CliffordThrust& src);


  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  void allocate(const uint64_t nqubit);

  //initialize
  void initialize(const uint64_t nqubit);

  // Get number of qubits of the CliffordThrust table
  uint64_t num_qubits() const {return num_qubits_;}

  // Return true if the number of qubits is 0
  bool empty() const {return (num_qubits_ == 0);}

  // Return JSON serialization of QubitVector;
  json_t json();

  // Access stabilizer table
//  Pauli::Pauli<BV::BinaryMatrixThrust> &operator[](uint64_t j) {return table_[j];}
//  const Pauli::Pauli<BV::BinaryMatrixThrust>& operator[](uint64_t j) const {return table_[j];}

  //set stabilizer
  void set_destabilizer(const int i, const Pauli::Pauli<BV::BinaryVector>& P);
  void set_stabilizer(const int i, const Pauli::Pauli<BV::BinaryVector>& P);

  //set phase
  void set_destabilizer_phases(const int i, const bool p);
  void set_stabilizer_phases(const int i, const bool p);

  // Set the state of the simulator to a given CliffordThrust
  void apply_set_stabilizer(const Clifford &clifford);


  //-----------------------------------------------------------------------
  // Apply basic CliffordThrust gates
  //-----------------------------------------------------------------------

  // Apply Controlled-NOT (CX) gate
  void append_cx(const uint64_t qubit_ctrl, const uint64_t qubit_trgt);

  // Apply Hadamard (H) gate
  void append_h(const uint64_t qubit);

  // Apply Phase (S, square root of Z) gate
  void append_s(const uint64_t qubit);

  // Apply Pauli::Pauli<BV::BinaryMatrixThrust> X gate
  void append_x(const uint64_t qubit);

  // Apply Pauli::Pauli<BV::BinaryMatrixThrust> Y gate
  void append_y(const uint64_t qubit);

  // Apply Pauli::Pauli<BV::BinaryMatrixThrust> Z gate
  void append_z(const uint64_t qubit);

  //-----------------------------------------------------------------------
  // Measurement
  //-----------------------------------------------------------------------

  // If we perform a single qubit Z measurement, 
  // will the outcome be random or deterministic.
  bool is_deterministic_outcome(const uint64_t& qubit);

  // Return the outcome (0 or 1) of a single qubit Z measurement, and
  // update the stabilizer to the conditional (post measurement) state if
  // the outcome was random.
  bool measure_and_update(const uint64_t qubit, const uint64_t randint);

  double expval_pauli(const reg_t &qubits,
                                const std::string& pauli);

  //-----------------------------------------------------------------------
  // Configuration settings
  //-----------------------------------------------------------------------

  // Set the threshold for chopping values to 0 in JSON
  void set_json_chop_threshold(double threshold);

  // Set the threshold for chopping values to 0 in JSON
  double get_json_chop_threshold() {return json_chop_threshold_;}

  // Set the maximum number of OpenMP thread for operations.
  void set_omp_threads(int n);

  // Get the maximum number of OpenMP thread for operations.
  uint64_t get_omp_threads() {return omp_threads_;}

  // Set the qubit threshold for activating OpenMP.
  // If self.qubits() > threshold OpenMP will be activated.
  void set_omp_threshold(int n);

  // Get the qubit threshold for activating OpenMP.
  uint64_t get_omp_threshold() {return omp_threshold_;}

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  Pauli::Pauli<BV::BinaryMatrixThrust> destabilizer_table_;
  Pauli::Pauli<BV::BinaryMatrixThrust> stabilizer_table_;
  BV::BinaryVectorThrust destabilizer_phases_;
  BV::BinaryVectorThrust stabilizer_phases_;
  uint64_t num_qubits_ = 0;

  uint_t device_id_;
  cudaStream_t stream_ = nullptr;
  AERDeviceVector<uint_t> reduce_buffer_;

  //buffers for accumulation of exponents
  AERDeviceVector<uint_t> exponent_buffer_;

  AERDeviceVector<uint_t> params_buffer_;

  std::vector<CliffordOp> ops_buffer_;
  AERDeviceVector<CliffordOp> ops_buffer_dev_;
  uint_t num_buffered_ops_;

  //-----------------------------------------------------------------------
  // Config settings
  //-----------------------------------------------------------------------

  uint64_t omp_threads_ = 1;          // Disable multithreading by default
  uint64_t omp_threshold_ = 1000;     // Qubit threshold for multithreading when enabled
  double json_chop_threshold_ = 0;  // Threshold for chopping small values
                                    // in JSON serialization

  //-----------------------------------------------------------------------
  // Helper functions
  //-----------------------------------------------------------------------

  // Check if there exists stabilizer or destabilizer row anticommuting
  // with Z[qubit]. If so return pair (true, row), else return (false, 0)
  std::pair<bool, uint64_t> z_anticommuting(const uint64_t qubit) const;

  // Check if there exists stabilizer or destabilizer row anticommuting
  // with X[qubit]. If so return pair (true, row), else return (false, 0)
  std::pair<bool, uint64_t> x_anticommuting(const uint64_t qubit) const;

  //run kernel
  template <typename Function>
  void apply_kernel(Function func, const uint_t count) const;

  template <typename Function>
  uint_t CliffordThrust::apply_kernel_reduce(Function func, const uint_t initial_val, const uint_t count) const;

  template <typename Function>
  uint_t apply_kernel_reduce_2D(Function func, const uint_t initial_val, const uint_t cols, const uint_t rows, const uint_t col_bits) const;

  uint_t* reduce_buffer(void) const
  {
    return ((uint_t*)thrust::raw_pointer_cast(reduce_buffer_.data()));
  }
  uint_t* exponent_buffer(void) const
  {
    return ((uint_t*)thrust::raw_pointer_cast(exponent_buffer_.data()));
  }
  uint_t* params_pointer(void) const
  {
    return ((uint_t*)thrust::raw_pointer_cast(params_buffer_.data()));
  }

  CliffordOp* ops_buffer_pointer(void) const
  {
    return ((CliffordOp*)thrust::raw_pointer_cast(ops_buffer_dev_.data()));
  }

  //apply buffered Clifford operations
  void apply_buffered_ops(void);
};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Config settings
//------------------------------------------------------------------------------

void CliffordThrust::set_json_chop_threshold(double threshold) {
  json_chop_threshold_ = threshold;
}

void CliffordThrust::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;
}

void CliffordThrust::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
}

//------------------------------------------------------------------------------
// run kernel code
//------------------------------------------------------------------------------
template <typename kernel_t> __global__
void cuda_apply_kernel(kernel_t func, uint_t count)
{
  uint_t i;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < count){
    func(i);
  }
}

template <typename kernel_t> __global__
void cuda_apply_kernel_reduce(uint_t* pReduceBuffer, kernel_t func, const uint_t initial_val, const uint_t count)
{
  __shared__ uint_t cache[32];
  uint_t sum;
  uint_t i,j,nw;

  i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < count)
    sum = func(i);
  else
    sum = initial_val;

  //reduce in warp
  nw = min(blockDim.x,warpSize);
  for(j=1;j<nw;j*=2){
    sum = func.reduce(sum, __shfl_xor_sync(0xffffffff,sum,j,32));
  }

  if(blockDim.x > warpSize){
    //reduce in thread block
    if((threadIdx.x & 31) == 0){
      cache[(threadIdx.x >> 5)] = sum;
    }
    __syncthreads();
    if(threadIdx.x < 32){
      if(threadIdx.x < ((blockDim.x+warpSize-1) >> 5))
        sum = cache[threadIdx.x];
      else
        sum = 0;

      //reduce in warp
      nw = warpSize;
      for(j=1;j<nw;j*=2){
        sum = func.reduce(sum, __shfl_xor_sync(0xffffffff,sum,j,32));
      }
    }
  }
  if(threadIdx.x == 0){
    pReduceBuffer[blockIdx.x] = sum;
  }
}

//kernel with reduction for 2D parallelization
template <typename kernel_t> __global__
void cuda_apply_kernel_reduce_2D(uint_t* pReduceBuffer, kernel_t func, const uint_t initial_val, const uint_t cols, const uint_t rows, const uint_t col_bits)
{
  __shared__ uint_t cache[1024];
  uint_t sum;
  uint_t i,j,nw;

  i = threadIdx.x + blockIdx.x * blockDim.x;
  uint_t ic = i & ((1ull << col_bits) - 1);
  uint_t ir = i >> col_bits;
  sum = func(ic, ir, cols, cache + ((threadIdx.x >> col_bits) << col_bits));

  //reduce in warp
  nw = min(blockDim.x,warpSize);
  for(j=1;j<nw;j*=2){
    sum = func.reduce(sum, __shfl_xor_sync(0xffffffff,sum,j,32));
  }

  if(blockDim.x > warpSize){
    //reduce in thread block
    if((threadIdx.x & 31) == 0){
      cache[(threadIdx.x >> 5)] = sum;
    }
    __syncthreads();
    if(threadIdx.x < 32){
      if(threadIdx.x < ((blockDim.x+warpSize-1) >> 5))
        sum = cache[threadIdx.x];
      else
        sum = 0;

      //reduce in warp
      nw = warpSize;
      for(j=1;j<nw;j*=2){
        sum = func.reduce(sum, __shfl_xor_sync(0xffffffff,sum,j,32));
      }
    }
  }
  if(threadIdx.x == 0){
    pReduceBuffer[blockIdx.x] = sum;
  }
}

template <typename kernel_t> __global__
void cuda_reduce_sum(uint_t *pReduceBuffer, kernel_t func, const uint_t initial_val, const uint_t n)
{
  __shared__ uint_t cache[32];
  uint_t sum;
  uint_t i,j,nw;

  i = threadIdx.x + blockIdx.x * blockDim.x;

  if(i < n)
    sum = pReduceBuffer[i];
  else
    sum = initial_val;

  //reduce in warp
  nw = min(blockDim.x,warpSize);
  for(j=1;j<nw;j*=2){
    sum = func.reduce(sum, __shfl_xor_sync(0xffffffff,sum,j,32));
  }

  if(blockDim.x > warpSize){
    //reduce in thread block
    if((threadIdx.x & 31) == 0){
      cache[(threadIdx.x >> 5)] = sum;
    }
    __syncthreads();
    if(threadIdx.x < 32){
      if(threadIdx.x < ((blockDim.x+warpSize-1) >> 5))
        sum = cache[threadIdx.x];
      else
        sum = 0.0;

      //reduce in warp
      nw = warpSize;
      for(j=1;j<nw;j*=2){
        sum = func.reduce(sum, __shfl_xor_sync(0xffffffff,sum,j,32));
      }
    }
  }
  if(threadIdx.x == 0){
    pReduceBuffer[blockIdx.x] = sum;
  }
}

template <typename Function>
void CliffordThrust::apply_kernel(Function func, const uint_t count) const
{
#ifdef AER_THRUST_CUDA
  uint_t nt,nb;
  nb = 1;
  nt = count;

  if(nt > 0){
    if(nt > QV_CUDA_NUM_THREADS){
      nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
      nt = QV_CUDA_NUM_THREADS;
    }
    cudaSetDevice(device_id_);
    cuda_apply_kernel<Function><<<nb,nt,0,stream_>>>(func, count);
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    std::stringstream str;
    str << "CliffordThrust::apply_kernel " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }

#else
  auto ci = thrust::counting_iterator<uint_t>(0);
  if(omp_threads_ > 1)
    thrust::for_each_n(thrust::device, ci , count, func);
  else
    thrust::for_each_n(thrust::seq, ci , count, func);
#endif
}

template <typename Function>
uint_t CliffordThrust::apply_kernel_reduce(Function func, const uint_t initial_val, const uint_t count) const
{
#ifdef AER_THRUST_CUDA
  uint_t n,nt,nb;
  nb = 1;
  nt = count;

  if(nt > 0){
    if(nt > QV_CUDA_NUM_THREADS){
      nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
      nt = QV_CUDA_NUM_THREADS;
    }
    else if(nt < 32){
      nt = 32;
    }
    cudaSetDevice(device_id_);
    cuda_apply_kernel_reduce<Function><<<nb,nt,0,stream_>>>(reduce_buffer(), func, initial_val, count);
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    std::stringstream str;
    str << "CliffordThrust::apply_kernel_reduce " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }
  while(nb > 1){
    n = nb;
    nt = nb;
    nb = 1;
    if(nt > QV_CUDA_NUM_THREADS){
      nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
      nt = QV_CUDA_NUM_THREADS;
    }
    else if(nt < 32){
      nt = 32;
    }
    cuda_reduce_sum<<<nb,nt,0,stream_>>>(reduce_buffer(), func, initial_val, n);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
      std::stringstream str;
      str << "CliffordThrust::apply_kernel_reduce sum " << cudaGetErrorName(err);
      throw std::runtime_error(str.str());
    }
  }
  uint_t ret;
  err = cudaMemcpyAsync(&ret,reduce_buffer(),sizeof(uint_t),cudaMemcpyDeviceToHost,stream_);
  if(err != cudaSuccess){
    std::stringstream str;
    str << "CliffordThrust::apply_kernel_reduce in cudaMemcpyAsync " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }
  err = cudaStreamSynchronize(stream_);
  if(err != cudaSuccess){
    std::stringstream str;
    str << "CliffordThrust::apply_kernel_reduce in cudaStreamSynchronize " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }

#else
  auto ci = thrust::counting_iterator<uint_t>(0);
  if(omp_threads_ > 1)
    thrust::for_each_n(thrust::device, ci , count, func);
  else
    thrust::for_each_n(thrust::seq, ci , count, func);
#endif
  return ret;
}

template <typename Function>
uint_t CliffordThrust::apply_kernel_reduce_2D(Function func, const uint_t initial_val, const uint_t cols, const uint_t rows, const uint_t col_bits) const
{
#ifdef AER_THRUST_CUDA
  uint_t n,nt,nb;
  nb = 1;
  nt = rows << col_bits;

  if(nt > 0){
    if(nt > QV_CUDA_NUM_THREADS){
      nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
      nt = QV_CUDA_NUM_THREADS;
    }
    else if(nt < 32){
      nt = 32;
    }
    cudaSetDevice(device_id_);
    cuda_apply_kernel_reduce_2D<Function><<<nb,nt,0,stream_>>>(reduce_buffer(), func, initial_val, cols, rows, col_bits);
  }
  cudaError_t err = cudaGetLastError();
  if(err != cudaSuccess){
    std::stringstream str;
    str << "CliffordThrust::apply_kernel_reduce " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }
  while(nb > 1){
    n = nb;
    nt = nb;
    nb = 1;
    if(nt > QV_CUDA_NUM_THREADS){
      nb = (nt + QV_CUDA_NUM_THREADS - 1) / QV_CUDA_NUM_THREADS;
      nt = QV_CUDA_NUM_THREADS;
    }
    else if(nt < 32){
      nt = 32;
    }
    cuda_reduce_sum<<<nb,nt,0,stream_>>>(reduce_buffer(), func, initial_val, n);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
      std::stringstream str;
      str << "CliffordThrust::apply_kernel_reduce sum " << cudaGetErrorName(err);
      throw std::runtime_error(str.str());
    }
  }
  uint_t ret;
  err = cudaMemcpyAsync(&ret,reduce_buffer(),sizeof(uint_t),cudaMemcpyDeviceToHost,stream_);
  if(err != cudaSuccess){
    std::stringstream str;
    str << "CliffordThrust::apply_kernel_reduce in cudaMemcpyAsync " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }
  err = cudaStreamSynchronize(stream_);
  if(err != cudaSuccess){
    std::stringstream str;
    str << "CliffordThrust::apply_kernel_reduce in cudaStreamSynchronize " << cudaGetErrorName(err);
    throw std::runtime_error(str.str());
  }

#else
  auto ci = thrust::counting_iterator<uint_t>(0);
  if(omp_threads_ > 1)
    thrust::for_each_n(thrust::device, ci , count, func);
  else
    thrust::for_each_n(thrust::seq, ci , count, func);
#endif
  return ret;
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

void CliffordThrust::allocate(const uint64_t nqubit)
{
  num_qubits_ = nqubit;

  int num_devices = 0;
  device_id_ = 0;
  if(cudaGetDeviceCount(&num_devices) == cudaSuccess){
    device_id_ = omp_get_thread_num() % num_devices;    //distribute shots
  }

  cudaSetDevice(device_id_);
  if(stream_){
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

  if(num_qubits_ < 1024)
    reduce_buffer_.resize(1);
  else
    reduce_buffer_.resize((num_qubits_ + 1023)/1024);

  destabilizer_table_.X.setLength(num_qubits_);
  destabilizer_table_.Z.setLength(num_qubits_);

  stabilizer_table_.X.setLength(num_qubits_);
  stabilizer_table_.Z.setLength(num_qubits_);

  // Add phases
  destabilizer_phases_.setLength(num_qubits_);
  stabilizer_phases_.setLength(num_qubits_);

  //allocate buffers for exponents
  exponent_buffer_.resize(destabilizer_phases_.blockLength()*6);

  ops_buffer_.resize(CLIFFORD_OPS_BUFFER_SIZE);
  ops_buffer_dev_.resize(CLIFFORD_OPS_BUFFER_SIZE);

  params_buffer_.resize(CLIFFORD_PARAMS_BUFFER_SIZE);

  num_buffered_ops_ = 0;
}

CliffordThrust::CliffordThrust(CliffordThrust& src)
{
  if(num_qubits_ != src.num_qubits_)
    allocate(src.num_qubits_);

  //copy state
  destabilizer_table_.X.copy_matrix(src.destabilizer_table_.X);
  destabilizer_table_.Z.copy_matrix(src.destabilizer_table_.Z);
  stabilizer_table_.X.copy_matrix(src.stabilizer_table_.X);
  stabilizer_table_.Z.copy_matrix(src.stabilizer_table_.Z);
  destabilizer_phases_.copy_vector(destabilizer_phases_);
  stabilizer_phases_.copy_vector(stabilizer_phases_);
}

CliffordThrust& CliffordThrust::operator=(CliffordThrust& src)
{
  //copy state
  destabilizer_table_.X.copy_matrix(src.destabilizer_table_.X);
  destabilizer_table_.Z.copy_matrix(src.destabilizer_table_.Z);
  stabilizer_table_.X.copy_matrix(src.stabilizer_table_.X);
  stabilizer_table_.Z.copy_matrix(src.stabilizer_table_.Z);
  destabilizer_phases_.copy_vector(destabilizer_phases_);
  stabilizer_phases_.copy_vector(stabilizer_phases_);
}

class init_kernel {
protected:
  uint_t* dX;
  uint_t* dZ;
  uint_t* sX;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
  uint_t offset;
  uint_t block_bits;
public:
  init_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp)
  {
    dX = dt.X.pointer(0);
    dZ = dt.Z.pointer(0);
    sX = st.X.pointer(0);
    sZ = st.Z.pointer(0);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
    offset = dt.X.blockLength();
    block_bits = dt.X.BLOCK_BITS;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t ii,q;
    ii = i % offset;
    q = i / offset;

    uint_t bits = 0;
    if((q >> block_bits) == ii){
      bits = 1ull << (q & ((1 << block_bits)-1));
    }
    dX[i] = bits;
    dZ[i] = 0;
    sX[i] = 0;
    sZ[i] = bits;

    if(q == 0){
      d_phase[ii] = 0;
      s_phase[ii] = 0;
    }
  }
};

void CliffordThrust::initialize(uint64_t nq)
{
  allocate(nq);

  apply_kernel(init_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_), nq*destabilizer_table_.X.blockLength());
}

CliffordThrust::~CliffordThrust()
{
  if(stream_){
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
  }
  reduce_buffer_.clear();
  reduce_buffer_.shrink_to_fit();
  exponent_buffer_.clear();
  exponent_buffer_.shrink_to_fit();
  ops_buffer_dev_.clear();
  ops_buffer_dev_.shrink_to_fit();
  params_buffer_.clear();
  params_buffer_.shrink_to_fit();
}

//------------------------------------------------------------------------------
// Apply CliffordThrust gates
//------------------------------------------------------------------------------
class cx_kernel {
protected:
  uint_t* dX_con;
  uint_t* dX_tar;
  uint_t* dZ_con;
  uint_t* dZ_tar;
  uint_t* sX_con;
  uint_t* sX_tar;
  uint_t* sZ_con;
  uint_t* sZ_tar;
  uint_t* d_phase;
  uint_t* s_phase;
public:
  cx_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t qcon, uint_t qtar)
  {
    dX_con = dt.X.pointer(qcon);
    dX_tar = dt.X.pointer(qtar);
    dZ_con = dt.Z.pointer(qcon);
    dZ_tar = dt.Z.pointer(qtar);
    sX_con = st.X.pointer(qcon);
    sX_tar = st.X.pointer(qtar);
    sZ_con = st.Z.pointer(qcon);
    sZ_tar = st.Z.pointer(qtar);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    const uint_t mask = (~0ull);

    d_phase[i] ^= (dX_con[i] & dZ_tar[i] & (dX_tar[i] ^ dZ_con[i] ^ mask));
    s_phase[i] ^= (sX_con[i] & sZ_tar[i] & (sX_tar[i] ^ sZ_con[i] ^ mask));

    dX_tar[i] = dX_tar[i] ^ dX_con[i];
    dZ_con[i] = dZ_tar[i] ^ dZ_con[i];
    sX_tar[i] = sX_tar[i] ^ sX_con[i];
    sZ_con[i] = sZ_tar[i] ^ sZ_con[i];
  }
};


void CliffordThrust::append_cx(const uint64_t qcon, const uint64_t qtar)
{
//  apply_kernel(cx_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, qcon, qtar), destabilizer_phases_.blockLength());
  if(num_buffered_ops_ >= ops_buffer_.size())
    apply_buffered_ops();
  ops_buffer_[num_buffered_ops_].op = 'c';
  ops_buffer_[num_buffered_ops_].tar = qtar;
  ops_buffer_[num_buffered_ops_].con = qcon;
  num_buffered_ops_++;
}

class h_kernel {
protected:
  uint_t* dX;
  uint_t* dZ;
  uint_t* sX;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
public:
  h_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t qubit)
  {
    dX = dt.X.pointer(qubit);
    dZ = dt.Z.pointer(qubit);
    sX = st.X.pointer(qubit);
    sZ = st.Z.pointer(qubit);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t t;

    d_phase[i] ^= (dX[i] & dZ[i]);
    s_phase[i] ^= (sX[i] & sZ[i]);
    // exchange X and Z
    t = dX[i];
    dX[i] = dZ[i];
    dZ[i] = t;
    t = sX[i];
    sX[i] = sZ[i];
    sZ[i] = t;
  }
};

void CliffordThrust::append_h(const uint64_t qubit) 
{
//  apply_kernel(h_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, qubit), destabilizer_phases_.blockLength());
  if(num_buffered_ops_ >= ops_buffer_.size())
    apply_buffered_ops();
  ops_buffer_[num_buffered_ops_].op = 'h';
  ops_buffer_[num_buffered_ops_].tar = qubit;
  num_buffered_ops_++;
}

class s_kernel {
protected:
  uint_t* dX;
  uint_t* dZ;
  uint_t* sX;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
public:
  s_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t qubit)
  {
    dX = dt.X.pointer(qubit);
    dZ = dt.Z.pointer(qubit);
    sX = st.X.pointer(qubit);
    sZ = st.Z.pointer(qubit);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    d_phase[i] ^= (dX[i] & dZ[i]);
    s_phase[i] ^= (sX[i] & sZ[i]);
    dZ[i] ^= dX[i];
    sZ[i] ^= sX[i];
  }
};

void CliffordThrust::append_s(const uint64_t qubit)
{
//  apply_kernel(s_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, qubit), destabilizer_phases_.blockLength());
  if(num_buffered_ops_ >= ops_buffer_.size())
    apply_buffered_ops();
  ops_buffer_[num_buffered_ops_].op = 's';
  ops_buffer_[num_buffered_ops_].tar = qubit;
  num_buffered_ops_++;
}

class x_kernel {
protected:
  uint_t* dZ;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
public:
  x_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t qubit)
  {
    dZ = dt.Z.pointer(qubit);
    sZ = st.Z.pointer(qubit);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    d_phase[i] ^= dZ[i];
    s_phase[i] ^= sZ[i];
  }
};

void CliffordThrust::append_x(const uint64_t qubit) 
{
//  apply_kernel(x_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, qubit), destabilizer_phases_.blockLength());
  if(num_buffered_ops_ >= ops_buffer_.size())
    apply_buffered_ops();
  ops_buffer_[num_buffered_ops_].op = 'x';
  ops_buffer_[num_buffered_ops_].tar = qubit;
  num_buffered_ops_++;
}

class y_kernel {
protected:
  uint_t* dX;
  uint_t* dZ;
  uint_t* sX;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
public:
  y_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t qubit)
  {
    dX = dt.X.pointer(qubit);
    dZ = dt.Z.pointer(qubit);
    sX = st.X.pointer(qubit);
    sZ = st.Z.pointer(qubit);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    d_phase[i] ^= (dZ[i] ^ dX[i]);
    s_phase[i] ^= (sZ[i] ^ sX[i]);
  }
};

void CliffordThrust::append_y(const uint64_t qubit) 
{
//  apply_kernel(y_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, qubit), destabilizer_phases_.blockLength());
  if(num_buffered_ops_ >= ops_buffer_.size())
    apply_buffered_ops();
  ops_buffer_[num_buffered_ops_].op = 'y';
  ops_buffer_[num_buffered_ops_].tar = qubit;
  num_buffered_ops_++;
}

class z_kernel {
protected:
  uint_t* dX;
  uint_t* sX;
  uint_t* d_phase;
  uint_t* s_phase;
public:
  z_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t qubit)
  {
    dX = dt.X.pointer(qubit);
    sX = st.X.pointer(qubit);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    d_phase[i] ^= dX[i];
    s_phase[i] ^= sX[i];
  }
};

void CliffordThrust::append_z(const uint64_t qubit) 
{
//  apply_kernel(z_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, qubit), destabilizer_phases_.blockLength());
  if(num_buffered_ops_ >= ops_buffer_.size())
    apply_buffered_ops();
  ops_buffer_[num_buffered_ops_].op = 'z';
  ops_buffer_[num_buffered_ops_].tar = qubit;
  num_buffered_ops_++;
}

//apply ops
class ops_kernel {
protected:
  uint_t* dX;
  uint_t* dZ;
  uint_t* sX;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
  CliffordOp* ops;
  uint_t num_ops;
  uint_t offset;
public:
  ops_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, CliffordOp* pOps, uint_t nops)
  {
    dX = dt.X.pointer(0);
    dZ = dt.Z.pointer(0);
    sX = st.X.pointer(0);
    sZ = st.Z.pointer(0);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
    ops = pOps;
    num_ops = nops;
    offset = dt.X.blockLength();
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    uint_t dp = d_phase[i];
    uint_t sp = s_phase[i];

    for(uint_t iop=0;iop<num_ops;iop++){
      uint_t tar = ops[iop].tar * offset + i;
      uint_t con = ops[iop].con * offset + i;

      if(ops[iop].op == 'c'){
        const uint_t mask = (~0ull);

        dp ^= (dX[con] & dZ[tar] & (dX[tar] ^ dZ[con] ^ mask));
        sp ^= (sX[con] & sZ[tar] & (sX[tar] ^ sZ[con] ^ mask));

        dX[tar] = dX[tar] ^ dX[con];
        dZ[con] = dZ[tar] ^ dZ[con];
        sX[tar] = sX[tar] ^ sX[con];
        sZ[con] = sZ[tar] ^ sZ[con];
      }
      else if(ops[iop].op == 'h'){
        uint_t t;

        dp ^= (dX[tar] & dZ[tar]);
        sp ^= (sX[tar] & sZ[tar]);
        // exchange X and Z
        t = dX[tar];
        dX[tar] = dZ[tar];
        dZ[tar] = t;
        t = sX[tar];
        sX[tar] = sZ[tar];
        sZ[tar] = t;
      }
      else if(ops[iop].op == 's'){
        dp ^= (dX[tar] & dZ[tar]);
        sp ^= (sX[tar] & sZ[tar]);
        dZ[tar] ^= dX[tar];
        sZ[tar] ^= sX[tar];
      }
      else if(ops[iop].op == 'x'){
        dp ^= dZ[tar];
        sp ^= sZ[tar];
      }
      else if(ops[iop].op == 'y'){
        dp ^= (dZ[tar] ^ dX[tar]);
        sp ^= (sZ[tar] ^ sX[tar]);
      }
      else if(ops[iop].op == 'z'){
        dp ^= dX[tar];
        sp ^= sX[tar];
      }
    }
    d_phase[i] = dp;
    s_phase[i] = sp;
  }

};

void CliffordThrust::apply_buffered_ops(void)
{
  if(num_buffered_ops_ == 0)
    return;

  cudaMemcpyAsync(ops_buffer_pointer(), ops_buffer_.data(), sizeof(CliffordOp)*num_buffered_ops_, cudaMemcpyHostToDevice, stream_);
  apply_kernel(ops_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, ops_buffer_pointer(), num_buffered_ops_), destabilizer_phases_.blockLength());

  cudaStreamSynchronize(stream_);
  num_buffered_ops_ = 0;
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------
class clear_anti_buffer_kernel {
protected:
  uint_t* save_ptr;
public:
  clear_anti_buffer_kernel(uint_t* param)
  {
    save_ptr = param;
  }

  __host__ __device__ void operator()(const uint_t &i) const
  {
    *save_ptr = ~(0ull);
  }
};

class anticommuting_kernel {
protected:
  uint_t* ptr;
  uint_t* save_ptr;
public:
  anticommuting_kernel(uint_t* p, uint_t* param)
  {
    ptr = p;
    save_ptr = param;
  }

#ifdef AER_THRUST_CUDA
  __device__ void operator()(const uint_t &i) const
#else
  __host__ __device__ uint_t operator()(const uint_t &i) const
#endif
  {
    uint_t s = ptr[i];
    uint_t ret = ~(0ull);
    if(s != 0){
      for (uint_t j = 0; j < 64; j++) {
        if(((s >> j) & 1) != 0){
          ret = ((i << 6) + j);
          break;
        }
      }
    }
#ifdef AER_THRUST_CUDA
    //use atomic fucntion to store result
    atomicMin((unsigned long long int*)save_ptr, ret);
#else
    return ret;
#endif
  }

  __host__ __device__ uint_t reduce(const uint_t r0, const uint_t r1) const
  {
    if(r0 <= r1)
      return r0;
    return r1;
  }
};

std::pair<bool, uint64_t> CliffordThrust::z_anticommuting(const uint64_t qubit) const 
{
  /*
  auto ci = thrust::counting_iterator<uint_t>(0);
  cudaStreamSynchronize(stream_);
  uint_t pos = thrust::transform_reduce(thrust::cuda::par.on(stream_), ci , ci + destabilizer_phases_.blockLength()
                                                      , anticommuting_kernel(stabilizer_table_.X.pointer(qubit)), (0ull - 1ull), thrust::minimum<uint_t>() );
  */
#ifdef AER_THRUST_CUDA
  uint_t pos = 0;   //result is not returned to host
  apply_kernel(clear_anti_buffer_kernel(params_pointer()), destabilizer_phases_.blockLength());
  apply_kernel(anticommuting_kernel(stabilizer_table_.X.pointer(qubit), params_pointer()), destabilizer_phases_.blockLength());
#else
  uint_t pos = apply_kernel_reduce(anticommuting_kernel(stabilizer_table_.X.pointer(qubit)), ~0ull, destabilizer_phases_.blockLength());
#endif
  if(pos < num_qubits_)
    return std::make_pair(true, pos);
  return std::make_pair(false, 0);
}


std::pair<bool, uint64_t> CliffordThrust::x_anticommuting(const uint64_t qubit) const 
{
  /*
  auto ci = thrust::counting_iterator<uint_t>(0);
  uint_t pos = thrust::transform_reduce(thrust::cuda::par.on(stream_), ci , ci + destabilizer_phases_.blockLength()
                                                      , anticommuting_kernel(stabilizer_table_.Z.pointer(qubit)), (0ull - 1ull), thrust::minimum<uint_t>() );
  */

#ifdef AER_THRUST_CUDA
  uint_t pos = 0;   //result is not returned to host
  apply_kernel(clear_anti_buffer_kernel(params_pointer()), destabilizer_phases_.blockLength());
  apply_kernel(anticommuting_kernel(stabilizer_table_.Z.pointer(qubit), params_pointer()), destabilizer_phases_.blockLength());
#else
  uint_t pos = apply_kernel_reduce(anticommuting_kernel(stabilizer_table_.Z.pointer(qubit)), ~0ull, destabilizer_phases_.blockLength());
#endif
  if(pos < num_qubits_)
    return std::make_pair(true, pos);
  return std::make_pair(false, 0);
}

void CliffordThrust::set_destabilizer(const int idx, const Pauli::Pauli<BV::BinaryVector>& P)
{
  for (int64_t i = 0; i < static_cast<int64_t>( num_qubits_); i++){
    destabilizer_table_.X.setValue(P.X[i], idx, i);
    destabilizer_table_.Z.setValue(P.Z[i], idx, i);
  }
}

void CliffordThrust::set_stabilizer(const int idx, const Pauli::Pauli<BV::BinaryVector>& P)
{
  for (int64_t i = 0; i < static_cast<int64_t>( num_qubits_); i++){
    stabilizer_table_.X.setValue(P.X[i], idx, i);
    stabilizer_table_.Z.setValue(P.Z[i], idx, i);
  }
}

void CliffordThrust::set_destabilizer_phases(const int i, const bool p)
{
  destabilizer_phases_.setValue(p, i);
}

void CliffordThrust::set_stabilizer_phases(const int i, const bool p)
{
  stabilizer_phases_.setValue(p, i);
}

void CliffordThrust::apply_set_stabilizer(const Clifford &clifford)
{
  for (int64_t i = 0; i < static_cast<int64_t>( num_qubits_); i++){
    destabilizer_table_.X.copy_vector(clifford.destabilizer_table_[i].X, i);
    destabilizer_table_.Z.copy_vector(clifford.destabilizer_table_[i].Z, i);
    stabilizer_table_.X.copy_vector(clifford.stabilizer_table_[i].X, i);
    stabilizer_table_.Z.copy_vector(clifford.stabilizer_table_[i].Z, i);
    destabilizer_phases_.copy_vector(clifford.destabilizer_phases_);
    stabilizer_phases_.copy_vector(clifford.stabilizer_phases_);
  }
}

//------------------------------------------------------------------------------
// Measurement
//------------------------------------------------------------------------------

bool CliffordThrust::is_deterministic_outcome(const uint64_t& qubit) {
  // CliffordThrust state measurements only have three probabilities:
  // (p0, p1) = (0.5, 0.5), (1, 0), or (0, 1)
  // The random case happens if there is a row anti-commuting with Z[qubit]
  apply_buffered_ops();
  return !z_anticommuting(qubit).first;
}

class measure_kernel {
protected:
  uint_t* dX;
  uint_t* dZ;
  uint_t* sX;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
  uint_t* exp_buffer;
  uint_t* z_anti;
  uint_t qubit;
  int block_bits;
  uint_t block_mask;
  uint_t offset;
  uint_t num_qubits;
  uint_t outcome;
public:
  measure_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t* buf, uint_t* p_anti, uint_t qubit_in, uint_t outcome_in)
  {
    dX = dt.X.pointer(0);
    dZ = dt.Z.pointer(0);
    sX = st.X.pointer(0);
    sZ = st.Z.pointer(0);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
    exp_buffer = buf;
    z_anti = p_anti;
    qubit = qubit_in;
    block_bits = dt.X.BLOCK_BITS;
    block_mask = dt.X.BLOCK_MASK;
    offset = dt.X.blockLength();
    num_qubits = dt.X.getLength();
    outcome = outcome_in;
  }

  __device__ uint_t operator()(const uint_t &ii, const uint_t &q, const uint_t &cols, uint_t* cache) const
  {
    uint_t ret = 0;
    uint_t row = *z_anti;
    if(row < num_qubits){   //non-deterministic
      non_deterministic_kernel(ii, q, row);
      if(ii == 0 && q == 0)
        ret = outcome*2;  //return outcome as a reduced result
    }
    else{     //deterministic
      ret = deterministic_kernel(ii, q, cols, cache);
    }
    return ret;
  }

  __device__ void non_deterministic_kernel(const uint_t &i, const uint_t &q, uint_t row) const
  {
    uint_t row_mask = ~0ull;

    if(i >= offset || q >= num_qubits)
      return;

    if((row >> block_bits) == i)
      row_mask ^= (1ull << (row & block_mask));

    uint_t d_mask = row_mask & dX[offset*qubit + i];
    uint_t s_mask = row_mask & sX[offset*qubit + i];

    if(d_mask != 0 || s_mask != 0){
      //calculating exponents by 2-bits integer * 64-qubits at once
      uint_t d0, d1;
      uint_t s0, s1;

      uint_t t0,t1;
      uint_t rX = 0ull - ((sX[offset*q + (row >> block_bits)] >> (row & block_mask)) & 1);
      uint_t rZ = 0ull - ((sZ[offset*q + (row >> block_bits)] >> (row & block_mask)) & 1);

      //destabilizer
      t0 = dX[offset*q + i] & rZ;
      t1 = dZ[offset*q + i] ^ rX;

      d0 = t0;
      d1 = (t0 & t1);

      t0 = rX & dZ[offset*q + i];
      t1 = rZ ^ dX[offset*q + i];
      t1 ^= t0;

      d1 ^= (t0 & d0);
      d0 ^= t0;
      d1 ^= (t0 & t1);

      dX[offset*q + i] ^= (d_mask & rX);
      dZ[offset*q + i] ^= (d_mask & rZ);

      //stabilizer
      t0 = sX[offset*q + i] & rZ;
      t1 = sZ[offset*q + i] ^ rX;

      s0 = t0;
      s1 = (t0 & t1);

      t0 = rX & sZ[offset*q + i];
      t1 = rZ ^ sX[offset*q + i];
      t1 ^= t0;

      s1 ^= (t0 & s0);
      s0 ^= t0;
      s1 ^= (t0 & t1);

      sX[offset*q + i] ^= (s_mask & rX);
      sZ[offset*q + i] ^= (s_mask & rZ);

      //reduce in exponent buffer
      unsigned long long int* d_exp_l = (unsigned long long int*)exp_buffer;
      unsigned long long int* d_exp_h = (unsigned long long int*)exp_buffer + offset;
      unsigned long long int* d_exp_c = (unsigned long long int*)exp_buffer + 2*offset;
      unsigned long long int* s_exp_l = (unsigned long long int*)exp_buffer + 3*offset;
      unsigned long long int* s_exp_h = (unsigned long long int*)exp_buffer + 4*offset;
      unsigned long long int* s_exp_c = (unsigned long long int*)exp_buffer + 5*offset;

      t0 = atomicXor(d_exp_l + i, (d0 & d_mask));
      atomicXor(d_exp_c + i, ((d0 & t0) & d_mask));      //add to carry bit if lower bit was 1
      atomicXor(d_exp_h + i, (d1 & d_mask));

      s0 = atomicXor(s_exp_l + i, (s0 & s_mask));
      atomicXor(s_exp_c + i, ((s0 & s0) & s_mask));      //add to carry bit if lower bit was 1
      atomicXor(s_exp_h + i, (s1 & s_mask));

      if(q == 0){ //update phases 
        uint_t rS = 0ull - (uint_t)((s_phase[row >> block_bits] >> (row & block_mask)) & 1);
        d_phase[i] ^= (rS & d_mask);
        s_phase[i] ^= (rS & s_mask);
      }
    }
  }

  __device__ uint_t deterministic_kernel(const uint_t &ii, const uint_t &q, const uint_t &cols, uint_t* cache) const
  {
    uint_t local_exponent_l = 0;
    uint_t local_exponent_h = 0;
    uint_t destabilizer_mask = 0;
    uint_t ax = 0;
    uint_t az = 0;
    uint_t sx = 0;
    uint_t sz = 0;
    uint_t accum = 0;
    uint_t block_size = 1 << block_bits;

    if(ii < offset && q < num_qubits){
      destabilizer_mask = dX[qubit*offset + ii];

      sx = sX[q*offset + ii];
      sz = sZ[q*offset + ii];

      //set accum for this block
      ax = destabilizer_mask & sx;
      az = destabilizer_mask & sz;
      for(int b=1;b<block_size;b*=2){
        ax ^= (ax << b);
        az ^= (az << b);
      }
      accum = (ax >> (block_size - 1)) | ( (az >> (block_size - 1)) << 1);
    }

    //accumulate all blocks
    uint_t c, cwarp = cols;
    if(cwarp > 32)
      cwarp = 32;

    //accumulate in warp
    uint_t t = accum;
    for(c=1;c<cwarp;c*=2){
      t = __shfl_xor_sync(0xffffffff,t,c,32);
      t ^= accum;
      if((ii ^ c) < ii)
        accum = t;
    }

    //if cols is larger than warp, use shared memory
    for(;c<cols;c*=2){
      cache[ii] = t;
      __syncthreads();
      uint_t pair = ii ^ c;
      t = cache[pair];
      t ^= accum;
      if(pair < ii)
        accum = t;
      __syncthreads();
      cache[ii] = t;
    }

    if(destabilizer_mask != 0){
      accum ^= (ax >> (block_size - 1)) | ( (az >> (block_size - 1)) << 1);
      ax ^= (0ull - (accum & 1));
      az ^= (0ull - ((accum >> 1) & 1));

      ax ^= sx;
      az ^= sz;

      ax &= destabilizer_mask;
      az &= destabilizer_mask;

      //exponents for this block
      uint_t t0,t1;

      t0 = ax & sz;
      t1 = az ^ sx;

      local_exponent_h ^= (t0 & local_exponent_l);
      local_exponent_l ^= t0;
      local_exponent_h ^= (t0 & t1);

      t0 = sx & az;
      t1 = sz ^ ax;
      t1 ^= t0;

      local_exponent_h ^= (t0 & local_exponent_l);
      local_exponent_l ^= t0;
      local_exponent_h ^= (t0 & t1);

      if(q == 0)
        local_exponent_h ^= (s_phase[ii] & destabilizer_mask);

      return AER::QV::Chunk::pop_count_kernel(local_exponent_l) + 2*AER::QV::Chunk::pop_count_kernel(local_exponent_h);
    }
    else
      return 0;
  }

  __host__ __device__ uint_t reduce(const uint_t r0, const uint_t r1) const
  {
    return r0 + r1;
  }
};

class measure_update_kernel {
protected:
  uint_t* dX;
  uint_t* dZ;
  uint_t* sX;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
  uint_t* exp_buffer;
  uint_t* z_anti;
  uint_t qubit;
  int block_bits;
  uint_t block_mask;
  uint_t offset;
  uint_t outcome;
  uint_t num_qubits;
public:
  measure_update_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t* buf, uint_t* p_anti, uint_t qubit_in, uint_t out_in)
  {
    dX = dt.X.pointer(0);
    dZ = dt.Z.pointer(0);
    sX = st.X.pointer(0);
    sZ = st.Z.pointer(0);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
    exp_buffer = buf;
    z_anti = p_anti;
    qubit = qubit_in;
    block_bits = dt.X.BLOCK_BITS;
    block_mask = dt.X.BLOCK_MASK;
    offset = dt.X.blockLength();
    outcome = 0ull - out_in;
    num_qubits = dt.X.getLength();
  }

  __host__ __device__ void operator()(const uint_t &q) const
  {
    uint_t row = *z_anti;
    if(row < num_qubits){   //non-deterministic
      uint_t pos = q*offset + (row >> block_bits);
      uint_t bit = row & block_mask;
      uint_t bit_mask = ~(1ull << bit);

      dX[pos] = (dX[pos] & bit_mask) | (sX[pos] & (~bit_mask));
      dZ[pos] = (dZ[pos] & bit_mask) | (sZ[pos] & (~bit_mask));
      sX[pos] &= bit_mask;
      sZ[pos] &= bit_mask;

      if((q & block_mask) == 0){
        //update phases if q is on top of bit block
        uint_t row_mask = ~0ull;
        uint_t i = q >> block_bits;
        if((row >> block_bits) == i)
          row_mask ^= (1ull << (row & block_mask));

        uint_t* d_exp_l = exp_buffer;
        uint_t* d_exp_h = exp_buffer + offset;
        uint_t* d_exp_c = exp_buffer + 2*offset;
        uint_t* s_exp_l = exp_buffer + 3*offset;
        uint_t* s_exp_h = exp_buffer + 4*offset;
        uint_t* s_exp_c = exp_buffer + 5*offset;

        uint_t d1;
        d1 = d_exp_h[i] ^ d_exp_c[i];
        d_phase[i] ^= d1;

        uint_t s1;
        s1 = s_exp_h[i] ^ s_exp_c[i];
        s_phase[i] ^= s1;

        //clear exponent buffers
        d_exp_l[i] = 0;
        d_exp_h[i] = 0;
        d_exp_c[i] = 0;
        s_exp_l[i] = 0;
        s_exp_h[i] = 0;
        s_exp_c[i] = 0;

        //update on row and qubit
        if(i == (row >> block_bits)){
          pos = qubit*offset + (row >> block_bits);
          sZ[pos] |= (~bit_mask);

          d_phase[i] = (d_phase[i] & bit_mask) | (s_phase[i] & (~bit_mask));
          s_phase[i] = (s_phase[i] & bit_mask) | (outcome & (~bit_mask));
        }
      }
    }
  }
};

class measure_non_determinisitic_kernel {
protected:
  uint_t* dX;
  uint_t* dZ;
  uint_t* sX;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
  uint_t* exp_buffer;
  uint_t row;
  uint_t qubit;
  int block_bits;
  uint_t block_mask;
  uint_t offset;
  uint_t num_qubits;
public:
  measure_non_determinisitic_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t* buf, uint_t qubit_in, uint_t row_in)
  {
    dX = dt.X.pointer(0);
    dZ = dt.Z.pointer(0);
    sX = st.X.pointer(0);
    sZ = st.Z.pointer(0);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
    exp_buffer = buf;
    row = row_in;
    qubit = qubit_in;
    block_bits = dt.X.BLOCK_BITS;
    block_mask = dt.X.BLOCK_MASK;
    offset = dt.X.blockLength();
    num_qubits = dt.X.getLength();
  }

  //atomic functions only can be called from device function
  __device__ void operator()(const uint_t &tid) const
  {
    uint_t row_mask = ~0ull;
    uint_t i, q;
    i = tid % offset;
    q = tid / offset;

    if((row >> block_bits) == i)
      row_mask ^= (1ull << (row & block_mask));

    uint_t d_mask = row_mask & dX[offset*qubit + i];
    uint_t s_mask = row_mask & sX[offset*qubit + i];

    if(d_mask != 0 || s_mask != 0){
      //calculating exponents by 2-bits integer * 64-qubits at once
      uint_t d0, d1;
      uint_t s0, s1;

      uint_t t0,t1;
      uint_t rX = 0ull - ((sX[offset*q + (row >> block_bits)] >> (row & block_mask)) & 1);
      uint_t rZ = 0ull - ((sZ[offset*q + (row >> block_bits)] >> (row & block_mask)) & 1);

      //destabilizer
      t0 = dX[offset*q + i] & rZ;
      t1 = dZ[offset*q + i] ^ rX;

      d0 = t0;
      d1 = (t0 & t1);

      t0 = rX & dZ[offset*q + i];
      t1 = rZ ^ dX[offset*q + i];
      t1 ^= t0;

      d1 ^= (t0 & d0);
      d0 ^= t0;
      d1 ^= (t0 & t1);

      dX[offset*q + i] ^= (d_mask & rX);
      dZ[offset*q + i] ^= (d_mask & rZ);

      //stabilizer
      t0 = sX[offset*q + i] & rZ;
      t1 = sZ[offset*q + i] ^ rX;

      s0 = t0;
      s1 = (t0 & t1);

      t0 = rX & sZ[offset*q + i];
      t1 = rZ ^ sX[offset*q + i];
      t1 ^= t0;

      s1 ^= (t0 & s0);
      s0 ^= t0;
      s1 ^= (t0 & t1);

      sX[offset*q + i] ^= (s_mask & rX);
      sZ[offset*q + i] ^= (s_mask & rZ);

      //reduce in exponent buffer
      unsigned long long int* d_exp_l = (unsigned long long int*)exp_buffer;
      unsigned long long int* d_exp_h = (unsigned long long int*)exp_buffer + offset;
      unsigned long long int* d_exp_c = (unsigned long long int*)exp_buffer + 2*offset;
      unsigned long long int* s_exp_l = (unsigned long long int*)exp_buffer + 3*offset;
      unsigned long long int* s_exp_h = (unsigned long long int*)exp_buffer + 4*offset;
      unsigned long long int* s_exp_c = (unsigned long long int*)exp_buffer + 5*offset;

      t0 = atomicXor(d_exp_l + i, (d0 & d_mask));
      atomicXor(d_exp_c + i, ((d0 & t0) & d_mask));      //add to carry bit if lower bit was 1
      atomicXor(d_exp_h + i, (d1 & d_mask));

      s0 = atomicXor(s_exp_l + i, (s0 & s_mask));
      atomicXor(s_exp_c + i, ((s0 & s0) & s_mask));      //add to carry bit if lower bit was 1
      atomicXor(s_exp_h + i, (s1 & s_mask));

      if(q == 0){ //update phases 
        uint_t rS = 0ull - (uint_t)((s_phase[row >> block_bits] >> (row & block_mask)) & 1);
        d_phase[i] ^= (rS & d_mask);
        s_phase[i] ^= (rS & s_mask);
      }
    }
  }
};

class measure_update_kernel_old {
protected:
  uint_t* dX;
  uint_t* dZ;
  uint_t* sX;
  uint_t* sZ;
  uint_t* d_phase;
  uint_t* s_phase;
  uint_t* exp_buffer;
  uint_t row;
  uint_t qubit;
  int block_bits;
  uint_t block_mask;
  uint_t offset;
  uint_t outcome;
public:
  measure_update_kernel_old(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& dp, BV::BinaryVectorThrust& sp, uint_t* buf, uint_t qubit_in, uint_t row_in, uint_t out_in)
  {
    dX = dt.X.pointer(0);
    dZ = dt.Z.pointer(0);
    sX = st.X.pointer(0);
    sZ = st.Z.pointer(0);
    d_phase = dp.pointer();
    s_phase = sp.pointer();
    exp_buffer = buf;
    row = row_in;
    qubit = qubit_in;
    block_bits = dt.X.BLOCK_BITS;
    block_mask = dt.X.BLOCK_MASK;
    offset = dt.X.blockLength();
    outcome = 0ull - out_in;
  }

  __host__ __device__ void operator()(const uint_t &q) const
  {
    uint_t pos = q*offset + (row >> block_bits);
    uint_t bit = row & block_mask;
    uint_t bit_mask = ~(1ull << bit);

    dX[pos] = (dX[pos] & bit_mask) | (sX[pos] & (~bit_mask));
    dZ[pos] = (dZ[pos] & bit_mask) | (sZ[pos] & (~bit_mask));
    sX[pos] &= bit_mask;
    sZ[pos] &= bit_mask;

    if((q & block_mask) == 0){
      //update phases if q is on top of bit block
      uint_t row_mask = ~0ull;
      uint_t i = q >> block_bits;
      if((row >> block_bits) == i)
        row_mask ^= (1ull << (row & block_mask));

      uint_t* d_exp_l = exp_buffer;
      uint_t* d_exp_h = exp_buffer + offset;
      uint_t* d_exp_c = exp_buffer + 2*offset;
      uint_t* s_exp_l = exp_buffer + 3*offset;
      uint_t* s_exp_h = exp_buffer + 4*offset;
      uint_t* s_exp_c = exp_buffer + 5*offset;

      uint_t d1;
      d1 = d_exp_h[i] ^ d_exp_c[i];
      d_phase[i] ^= d1;

      uint_t s1;
      s1 = s_exp_h[i] ^ s_exp_c[i];
      s_phase[i] ^= s1;

      //clear exponent buffers
      d_exp_l[i] = 0;
      d_exp_h[i] = 0;
      d_exp_c[i] = 0;
      s_exp_l[i] = 0;
      s_exp_h[i] = 0;
      s_exp_c[i] = 0;

      //update on row and qubit
      if(i == (row >> block_bits)){
        pos = qubit*offset + (row >> block_bits);
        sZ[pos] |= (~bit_mask);

        d_phase[i] = (d_phase[i] & bit_mask) | (s_phase[i] & (~bit_mask));
        s_phase[i] = (s_phase[i] & bit_mask) | (outcome & (~bit_mask));
      }
    }
  }
};

class measure_determinisitic_kernel_old {
protected:
  uint_t* dX;
  uint_t* sX;
  uint_t* sZ;
  uint_t* s_phase;
  uint_t row;
  uint_t qubit;
  int block_bits;
  uint_t block_mask;
  uint_t offset;
  uint_t block_size;
public:
  measure_determinisitic_kernel_old(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& sp, uint_t qubit_in)
  {
    dX = dt.X.pointer(0);
    sX = st.X.pointer(0);
    sZ = st.Z.pointer(0);
    s_phase = sp.pointer();
    qubit = qubit_in;
    block_bits = dt.X.BLOCK_BITS;
    block_mask = dt.X.BLOCK_MASK;
    offset = dt.X.blockLength();
    block_size = 1 << block_bits;
  }

  __host__ __device__ uint_t operator()(const uint_t &q) const
  {
    bool accumX = false;
    bool accumZ = false;

    uint_t local_exponent_l = 0;
    uint_t local_exponent_h = 0;
    for (uint_t ii = 0; ii < offset; ii++) {
      uint_t destabilizer_mask = dX[qubit*offset + ii];
      if(destabilizer_mask == 0)
        continue;

      uint_t sx = sX[q*offset + ii];
      uint_t sz = sZ[q*offset + ii];

      //set accum for this block
      uint_t ax = destabilizer_mask & sx;
      uint_t az = destabilizer_mask & sz;
      for(int b=1;b<block_size;b*=2){
        ax ^= (ax << b);
        az ^= (az << b);
      }
      ax ^= (0ull - (uint_t)accumX);
      az ^= (0ull - (uint_t)accumZ);

      accumX = (ax >> (block_size - 1)) == 1;
      accumZ = (az >> (block_size - 1)) == 1;

      ax ^= sx;
      az ^= sz;

      //exponents for this block
      uint_t t0,t1;

      t0 = ax & sz;
      t1 = az ^ sx;

      local_exponent_h ^= (t0 & local_exponent_l);
      local_exponent_l ^= t0;
      local_exponent_h ^= (t0 & t1);

      t0 = sx & az;
      t1 = sz ^ ax;
      t1 ^= t0;

      local_exponent_h ^= (t0 & local_exponent_l);
      local_exponent_l ^= t0;
      local_exponent_h ^= (t0 & t1);

      if(q == 0)
        local_exponent_h ^= (s_phase[ii] & destabilizer_mask);
    }
    return AER::QV::Chunk::pop_count_kernel(local_exponent_l) + 2*AER::QV::Chunk::pop_count_kernel(local_exponent_h);
  }

  __host__ __device__ uint_t reduce(const uint_t r0, const uint_t r1) const
  {
    return r0 + r1;
  }
};

class measure_determinisitic_kernel {
protected:
  uint_t* dX;
  uint_t* sX;
  uint_t* sZ;
  uint_t* s_phase;
  uint_t row;
  uint_t qubit;
  int block_bits;
  uint_t block_mask;
  uint_t offset;
  uint_t block_size;
  uint_t num_qubits;
public:
  measure_determinisitic_kernel(Pauli::Pauli<BV::BinaryMatrixThrust>& dt, Pauli::Pauli<BV::BinaryMatrixThrust>& st, BV::BinaryVectorThrust& sp, uint_t qubit_in, uint_t nqubits)
  {
    dX = dt.X.pointer(0);
    sX = st.X.pointer(0);
    sZ = st.Z.pointer(0);
    s_phase = sp.pointer();
    qubit = qubit_in;
    block_bits = dt.X.BLOCK_BITS;
    block_mask = dt.X.BLOCK_MASK;
    offset = dt.X.blockLength();
    block_size = 1 << block_bits;
    num_qubits = nqubits;
  }

  __device__ uint_t operator()(const uint_t &ii, const uint_t &q, const uint_t &cols, uint_t* cache) const
  {
    uint_t local_exponent_l = 0;
    uint_t local_exponent_h = 0;
    uint_t destabilizer_mask = 0;
    uint_t ax = 0;
    uint_t az = 0;
    uint_t sx = 0;
    uint_t sz = 0;
    uint_t accum = 0;

    if(ii < offset && q < num_qubits){
      destabilizer_mask = dX[qubit*offset + ii];

      sx = sX[q*offset + ii];
      sz = sZ[q*offset + ii];

      //set accum for this block
      ax = destabilizer_mask & sx;
      az = destabilizer_mask & sz;
      for(int b=1;b<block_size;b*=2){
        ax ^= (ax << b);
        az ^= (az << b);
      }
      accum = (ax >> (block_size - 1)) | ( (az >> (block_size - 1)) << 1);
    }

    //accumulate all blocks
    uint_t c, cwarp = cols;
    if(cwarp > 32)
      cwarp = 32;

    //accumulate in warp
    uint_t t = accum;
    for(c=1;c<cwarp;c*=2){
      t = __shfl_xor_sync(0xffffffff,t,c,32);
      t ^= accum;
      if((ii ^ c) < ii)
        accum = t;
    }

    //if cols is larger than warp, use shared memory
    for(;c<cols;c*=2){
      cache[ii] = t;
      __syncthreads();
      uint_t pair = ii ^ c;
      t = cache[pair];
      t ^= accum;
      if(pair < ii)
        accum = t;
      __syncthreads();
      cache[ii] = t;
    }

    if(destabilizer_mask != 0){
      accum ^= (ax >> (block_size - 1)) | ( (az >> (block_size - 1)) << 1);
      ax ^= (0ull - (accum & 1));
      az ^= (0ull - ((accum >> 1) & 1));

      ax ^= sx;
      az ^= sz;

      ax &= destabilizer_mask;
      az &= destabilizer_mask;

      //exponents for this block
      uint_t t0,t1;

      t0 = ax & sz;
      t1 = az ^ sx;

      local_exponent_h ^= (t0 & local_exponent_l);
      local_exponent_l ^= t0;
      local_exponent_h ^= (t0 & t1);

      t0 = sx & az;
      t1 = sz ^ ax;
      t1 ^= t0;

      local_exponent_h ^= (t0 & local_exponent_l);
      local_exponent_l ^= t0;
      local_exponent_h ^= (t0 & t1);

      if(q == 0)
        local_exponent_h ^= (s_phase[ii] & destabilizer_mask);

      return AER::QV::Chunk::pop_count_kernel(local_exponent_l) + 2*AER::QV::Chunk::pop_count_kernel(local_exponent_h);
    }
    else
      return 0;
  }

  __host__ __device__ uint_t reduce(const uint_t r0, const uint_t r1) const
  {
    return r0 + r1;
  }
};

bool CliffordThrust::measure_and_update(const uint64_t qubit, const uint64_t randint) 
{
  apply_buffered_ops();

  // Clifford state measurements only have three probabilities:
  // (p0, p1) = (0.5, 0.5), (1, 0), or (0, 1)
  // The random case happens if there is a row anti-commuting with Z[qubit]
  auto anticom = z_anticommuting(qubit);

  uint_t col_bits = 0;
  if(destabilizer_phases_.blockLength() > 1){
    uint_t bits = destabilizer_phases_.blockLength();
    uint_t count = 0;
    while(bits != 0){
      if((bits & 1) != 0)
        count++;
      bits >>= 1;
      col_bits++;
    }
    if(count == 1){
      col_bits--;
    }
  }

  bool outcome = (randint == 1);
  uint_t count = apply_kernel_reduce_2D(
                 measure_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, exponent_buffer(), params_pointer(), qubit, outcome),
                 0, 1ull << col_bits, num_qubits_, col_bits);
  apply_kernel(measure_update_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, exponent_buffer(), params_pointer(), qubit, outcome), num_qubits_);
  return ((count & 3) == 2);

  /*
  if (anticom.first) {
    bool outcome = (randint == 1);
    auto row = anticom.second;
    apply_kernel(measure_non_determinisitic_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, exponent_buffer(), qubit, row), destabilizer_phases_.blockLength()*num_qubits_);
    apply_kernel(measure_update_kernel(destabilizer_table_, stabilizer_table_, destabilizer_phases_, stabilizer_phases_, exponent_buffer(), qubit, row, outcome), num_qubits_);

    return outcome;
  }
  else {
    // Deterministic outcome
    uint_t col_bits = 0;
    if(destabilizer_phases_.blockLength() > 1){
      uint_t bits = destabilizer_phases_.blockLength();
      uint_t count = 0;
      while(bits != 0){
        if((bits & 1) != 0)
          count++;
        bits >>= 1;
        col_bits++;
      }
      if(count == 1){
        col_bits--;
      }
    }

    uint_t count = apply_kernel_reduce_2D(measure_determinisitic_kernel(destabilizer_table_, stabilizer_table_, stabilizer_phases_, qubit, num_qubits_), 0, 1ull << col_bits, num_qubits_, col_bits);
    return ((count & 3) == 2);
  }
  */
}

double CliffordThrust::expval_pauli(const reg_t &qubits,
                              const std::string& pauli)
{
  // Construct Pauli on N-qubits
  Pauli::Pauli<BV::BinaryVector> P(num_qubits_);
  uint_t phase = 0;
  for (size_t i = 0; i < qubits.size(); ++i) {
    switch (pauli[pauli.size() - 1 - i]) {
      case 'X':
        P.X.set1(qubits[i]);
        break;
      case 'Y':
        P.X.set1(qubits[i]);
        P.Z.set1(qubits[i]);
        phase += 1;
        break;
      case 'Z':
        P.Z.set1(qubits[i]);
        break;
      default:
        break;
    };
  }

  // Check if there is a stabilizer that anti-commutes with an odd number of qubits
  // If so expectation value is 0
  for (size_t i = 0; i < num_qubits_; i++) {
    size_t num_anti = 0;
    for (const auto& qubit : qubits) {
      if (P.Z[qubit] & stabilizer_table_.X.getValue(i, qubit)) {
	      num_anti++;
      }
      if (P.X[qubit] & stabilizer_table_.Z.getValue(i, qubit)) {
	      num_anti++;
      }
    }
    if(num_anti % 2 == 1)
      return 0.0;
  }

  // Otherwise P is (-1)^a prod_j S_j^b_j for CliffordThrust stabilizers
  // If P anti-commutes with D_j then b_j = 1.
  // Multiply P by stabilizers with anti-commuting destabilizers
  auto PZ = P.Z; // Make a copy of P.Z 
  for (size_t i = 0; i < num_qubits_; i++) {
    // Check if destabilizer anti-commutes
    size_t num_anti = 0;
    for (const auto& qubit : qubits) {
      if (P.Z[qubit] & destabilizer_table_.X.getValue(i, qubit)) {
	      num_anti++;
      }
      if (P.X[qubit] & destabilizer_table_.Z.getValue(i, qubit)) {
	      num_anti++;
      }
    }
    if (num_anti % 2 == 0) continue;

    // If anti-commutes multiply Pauli by stabilizer
    phase += 2 * (uint_t)stabilizer_phases_[i];
    for (size_t k = 0; k < num_qubits_; k++) {
      phase += stabilizer_table_.Z.getValue(i, k) & stabilizer_table_.X.getValue(i, k);
      phase += 2 * (PZ[k] & stabilizer_table_.X.getValue(i, k));
      PZ.setValue(PZ[k] ^ stabilizer_table_.Z.getValue(i, k), k);
    }
  }
  return (phase % 4) ? -1.0 : 1.0;
}


//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

json_t CliffordThrust::json() 
{
  json_t js = json_t::object();
  // Add destabilizers
  json_t stab;
  apply_buffered_ops();
  for (size_t i = 0; i < num_qubits_; i++) {
    // Destabilizer
    std::string label = (destabilizer_phases_[i] == 0) ? "+" : "-";

    Pauli::Pauli<BV::BinaryVector> P(num_qubits_);
    for (size_t j = 0; j < num_qubits_; j++) {
      P.X.setValue(destabilizer_table_.X.getValue(i, j), j);
      P.Z.setValue(destabilizer_table_.Z.getValue(i, j), j);
    }
    label += P.str();
    js["destabilizer"].push_back(label);

    // Stabilizer
    label = (stabilizer_phases_[i] == 0) ? "+" : "-";
    for (size_t j = 0; j < num_qubits_; j++) {
      P.X.setValue(stabilizer_table_.X.getValue(i, j), j);
      P.Z.setValue(stabilizer_table_.Z.getValue(i, j), j);
    }
    label += P.str();
    js["stabilizer"].push_back(label);
  }
  return js;
}

inline void to_json(json_t &js, CliffordThrust &clif) {
  js = clif.json();
}

//------------------------------------------------------------------------------
} // end namespace CliffordThrust
} // AER
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <class statevector_t>
std::ostream &operator<<(std::ostream &out, AER::Clifford::CliffordThrust &clif) {
  out << clif.json().dump();
  return out;
}

//------------------------------------------------------------------------------
#endif
