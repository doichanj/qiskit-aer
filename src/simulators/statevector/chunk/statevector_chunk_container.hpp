/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020, 2021, 2022.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */


#ifndef _qv_statevector_chunk_container_hpp_
#define _qv_statevector_chunk_container_hpp_

#include "simulators/statevector/chunk/device_chunk_container.hpp"

namespace AER {
namespace QV {
namespace Chunk {

//reserve 512MB of memory for Thrust internal use
#define RESERVE_FOR_THRUST (1ull << 28)

//============================================================================
// device chunk container class
//============================================================================
template <typename data_t>
class StatevectorChunkContainer : public DeviceChunkContainer<thrust::complex<data_t>>
{
protected:

public:
  StatevectorChunkContainer()
  {
  }
  ~StatevectorChunkContainer()
  {
    DeviceChunkContainer<thrust::complex<data_t>>::Deallocate();
  }

  double norm(uint_t iChunk,uint_t count) const override;
  double trace(uint_t iChunk,uint_t row,uint_t count) const override;
  reg_t sample_measure(uint_t iChunk,const std::vector<double> &rnds, uint_t stride = 1, bool dot = true,uint_t count = 1) const override;

  //apply matrix 
  void apply_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &mat,const uint_t gid, const uint_t count) override;

  //apply diagonal matrix
  void apply_diagonal_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &diag,const uint_t gid, const uint_t count) override;

  //apply (controlled) X
  void apply_X(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count) override;

  //apply (controlled) Y
  void apply_Y(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count) override;

  //apply (controlled) phase
  void apply_phase(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const std::complex<double> phase,const uint_t gid, const uint_t count) override;

  //apply (controlled) swap gate
  void apply_swap(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const uint_t gid, const uint_t count) override;

  //apply multiple swap gates
  void apply_multi_swaps(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count) override;

  //apply permutation
  void apply_permutation(const uint_t iChunk,const reg_t& qubits,const std::vector<std::pair<uint_t, uint_t>> &pairs, const uint_t gid, const uint_t count) override;

  //apply rotation around axis
  void apply_rotation(const uint_t iChunk,const reg_t &qubits, const Rotation r, const double theta, const uint_t gid, const uint_t count) override;

  //get probabilities of chunk
  void probabilities(std::vector<double>& probs, const uint_t iChunk, const reg_t& qubits) const override;

  //Pauli expectation values
  double expval_pauli(const uint_t iChunk,const reg_t& qubits,const std::string &pauli,const complex_t initial_phase) const override;


  //do all gates stored in queue
  void apply_blocked_gates(uint_t iChunk) override;
};

template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &mat,const uint_t gid, const uint_t count)
{
  const size_t N = qubits.size() - control_bits;

  if(N == 1){
    if(control_bits == 0)
      this->Execute(MatrixMult2x2<data_t>(mat,qubits[0]), iChunk, gid, count);
    else  //2x2 matrix with control bits
      this->Execute(MatrixMult2x2Controlled<data_t>(mat,qubits), iChunk, gid, count);
  }
  else if(N == 2){
    this->Execute(MatrixMult4x4<data_t>(mat,qubits[0],qubits[1]), iChunk, gid, count);
  }
  else{
    auto qubits_sorted = qubits;
    std::sort(qubits_sorted.begin(), qubits_sorted.end());
#ifndef AER_THRUST_CUDA
    if(N == 3){
      this->StoreMatrix(mat, iChunk);
      this->Execute(MatrixMult8x8<data_t>(qubits,qubits_sorted), iChunk, gid, count);
    }
    else if(N == 4){
      this->StoreMatrix(mat, iChunk);
      this->Execute(MatrixMult16x16<data_t>(qubits,qubits_sorted), iChunk, gid, count);
    }
    else if(N <= 10){
#else
    if(N <= 10){
#endif
      int i;
      for(i=0;i<N;i++){
        qubits_sorted.push_back(qubits[i]);
      }
      this->StoreMatrix(mat, iChunk);
      this->StoreUintParams(qubits_sorted, iChunk);

      this->Execute(MatrixMultNxN<data_t>(N), iChunk, gid, count);
    }
    else{
      cvector_t<double> matLU;
      reg_t params;
      MatrixMultNxN_LU<data_t> f(mat,qubits_sorted,matLU,params);

      this->StoreMatrix(matLU, iChunk);
      this->StoreUintParams(params, iChunk);

      this->Execute(f, iChunk, gid, count);
    }
  }
}

template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_diagonal_matrix(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const cvector_t<double> &diag,const uint_t gid, const uint_t count)
{
  const size_t N = qubits.size() - control_bits;

  if(N == 1){
    if(control_bits == 0)
      this->Execute(DiagonalMult2x2<data_t>(diag,qubits[0]), iChunk, gid, count);
    else
      this->Execute(DiagonalMult2x2Controlled<data_t>(diag,qubits), iChunk, gid, count);
  }
  else if(N == 2){
    this->Execute(DiagonalMult4x4<data_t>(diag,qubits[0],qubits[1]), iChunk, gid, count);
  }
  else{
    this->StoreMatrix(diag, iChunk);
    this->StoreUintParams(qubits, iChunk);

    this->Execute(DiagonalMultNxN<data_t>(qubits), iChunk, gid, count);
  }
}

template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_X(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count)
{
  this->Execute(CX_func<data_t>(qubits), iChunk, gid, count);
}

template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_Y(const uint_t iChunk,const reg_t& qubits,const uint_t gid, const uint_t count)
{
  this->Execute(CY_func<data_t>(qubits), iChunk, gid, count);
}

template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_phase(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const std::complex<double> phase,const uint_t gid, const uint_t count)
{
  this->Execute(phase_func<data_t>(qubits,*(thrust::complex<double>*)&phase), iChunk, gid, count );
}

template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_swap(const uint_t iChunk,const reg_t& qubits,const int_t control_bits,const uint_t gid, const uint_t count)
{
  this->Execute(CSwap_func<data_t>(qubits), iChunk, gid, count);
}


template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_multi_swaps(const uint_t iChunk,const reg_t& qubits,const uint_t gid,const uint_t count)
{
  //max 5 swaps can be applied at once using GPU's shared memory
  for(int_t i=0;i<qubits.size();i+=10){
    int_t n = 10;
    if(i + n > qubits.size())
      n = qubits.size() - i;

    reg_t qubits_swap(qubits.begin() + i,qubits.begin() + i + n);
    std::sort(qubits_swap.begin(), qubits_swap.end());
    qubits_swap.insert(qubits_swap.end(), qubits.begin() + i,qubits.begin() + i + n);

    this->StoreUintParams(qubits_swap, iChunk);
    this->Execute(MultiSwap_func<data_t>(n), iChunk, gid, count);
  }
}

template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_permutation(const uint_t iChunk,const reg_t& qubits,const std::vector<std::pair<uint_t, uint_t>> &pairs, const uint_t gid, const uint_t count)
{
  const size_t N = qubits.size();
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  reg_t params;
  Permutation<data_t> f(qubits_sorted,qubits,pairs,params);

  this->StoreUintParams(params, iChunk);

  this->Execute(f, iChunk, gid, count);
}

template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_rotation(const uint_t iChunk,const reg_t &qubits, const Rotation r, const double theta, const uint_t gid, const uint_t count)
{
  int control_bits = qubits.size() - 1;
  switch(r){
    case Rotation::x:
      apply_matrix(iChunk, qubits, control_bits, Linalg::VMatrix::rx(theta), gid, count);
      break;
    case Rotation::y:
      apply_matrix(iChunk, qubits, control_bits, Linalg::VMatrix::ry(theta), gid, count);
      break;
    case Rotation::z:
      apply_diagonal_matrix(iChunk, qubits, control_bits, Linalg::VMatrix::rz_diag(theta), gid, count);
      break;
    case Rotation::xx:
      apply_matrix(iChunk, qubits, control_bits-1, Linalg::VMatrix::rxx(theta), gid, count);
      break;
    case Rotation::yy:
      apply_matrix(iChunk, qubits, control_bits-1, Linalg::VMatrix::ryy(theta), gid, count);
      break;
    case Rotation::zz:
      apply_diagonal_matrix(iChunk, qubits, control_bits-1, Linalg::VMatrix::rzz_diag(theta), gid, count);
      break;
    case Rotation::zx:
      apply_matrix(iChunk, qubits, control_bits-1, Linalg::VMatrix::rzx(theta), gid, count);
      break;
    default:
      throw std::invalid_argument(
          "QubitVectorThrust::invalid rotation axis.");
  }
}

template <typename data_t>
void StatevectorChunkContainer<data_t>::probabilities(std::vector<double>& probs, const uint_t iChunk, const reg_t& qubits) const
{
  const size_t N = qubits.size();
  const int_t DIM = 1 << N;
  probs.resize(DIM);

  if(N == 1){ //special case for 1 qubit (optimized for measure)
    this->ExecuteSum2(&probs[0],probability_1qubit_func<data_t>(qubits[0]), iChunk, 1);
  }
  else{
    for(int_t i=0;i<DIM;i++){
      this->ExecuteSum(&probs[i],probability_func<data_t>(qubits,i), iChunk, 1);
    }
  }
}

template <typename data_t>
double StatevectorChunkContainer<data_t>::norm(uint_t iChunk,uint_t count) const
{
  double ret;
  this->ExecuteSum(&ret,norm_func<data_t>(), iChunk, count);

  return ret;
}

template <typename data_t>
double StatevectorChunkContainer<data_t>::trace(uint_t iChunk,uint_t row,uint_t count) const
{
  double ret;
  this->ExecuteSum(&ret,trace_func<data_t>(row), iChunk, count);

  return ret;
}

template <typename data_t>
reg_t StatevectorChunkContainer<data_t>::sample_measure(uint_t iChunk,const std::vector<double> &rnds, uint_t stride, bool dot,uint_t count) const
{
  const int_t SHOTS = rnds.size();
  reg_t samples(SHOTS,0);

  this->set_device();

  strided_range<thrust::complex<data_t>*> iter(this->chunk_pointer(iChunk), this->chunk_pointer(iChunk+count), stride);

#ifdef AER_THRUST_CUDA

  if(dot)
    thrust::transform_inclusive_scan(thrust::cuda::par.on(this->stream(iChunk)),iter.begin(),iter.end(),iter.begin(),complex_dot_scan<data_t>(),thrust::plus<thrust::complex<data_t>>());
  else
    thrust::inclusive_scan(thrust::cuda::par.on(this->stream(iChunk)),iter.begin(),iter.end(),iter.begin(),thrust::plus<thrust::complex<data_t>>());

  uint_t iBuf = 0;
  if(this->multi_shots_)
    iBuf = iChunk;

  double* pRnd = (double*)this->matrix_pointer(iBuf);
  uint_t* pSmp = this->param_pointer(iBuf);
  thrust::device_ptr<double> rnd_dev_ptr = thrust::device_pointer_cast(pRnd);
  uint_t i,nshots,size = this->matrix_.size()*2;
  if(size > this->params_.size())
    size = this->params_.size();

  for(i=0;i<SHOTS;i+=size){
    nshots = size;
    if(i + nshots > SHOTS)
      nshots = SHOTS - i;

    cudaMemcpyAsync(pRnd,&rnds[i],nshots*sizeof(double),cudaMemcpyHostToDevice,this->stream(iChunk));

    thrust::lower_bound(thrust::cuda::par.on(this->stream(iChunk)), iter.begin(), iter.end(), rnd_dev_ptr, rnd_dev_ptr + nshots, this->params_.begin() + (iBuf * this->params_buffer_size_),complex_less<data_t>());

    cudaMemcpyAsync(&samples[i],pSmp,nshots*sizeof(uint_t),cudaMemcpyDeviceToHost,this->stream(iChunk));
  }
  cudaStreamSynchronize(this->stream(iChunk));
#else
  if(this->omp_threads_ > 1){
    if(dot)
      thrust::transform_inclusive_scan(thrust::device,iter.begin(),iter.end(),iter.begin(),complex_dot_scan<data_t>(),thrust::plus<thrust::complex<data_t>>());
    else
      thrust::inclusive_scan(thrust::device,iter.begin(),iter.end(),iter.begin(),thrust::plus<thrust::complex<data_t>>());

    thrust::lower_bound(thrust::device, iter.begin(), iter.end(), rnds.begin(), rnds.begin() + SHOTS, samples.begin() ,complex_less<data_t>());
  }
  else{
    if(dot)
      thrust::transform_inclusive_scan(thrust::seq,iter.begin(),iter.end(),iter.begin(),complex_dot_scan<data_t>(),thrust::plus<thrust::complex<data_t>>());
    else
      thrust::inclusive_scan(thrust::seq,iter.begin(),iter.end(),iter.begin(),thrust::plus<thrust::complex<data_t>>());

    thrust::lower_bound(thrust::seq, iter.begin(), iter.end(), rnds.begin(), rnds.begin() + SHOTS, samples.begin() ,complex_less<data_t>());
  }
#endif

  return samples;
}

template <typename data_t>
double StatevectorChunkContainer<data_t>::expval_pauli(const uint_t iChunk,const reg_t& qubits,const std::string &pauli,const complex_t initial_phase) const
{
  uint_t x_mask, z_mask, num_y, x_max;
  std::tie(x_mask, z_mask, num_y, x_max) = pauli_masks_and_phase(qubits, pauli);

  // Special case for only I Paulis
  if (x_mask + z_mask == 0) {
    thrust::complex<double> ret = norm(iChunk, 1);
    return ret.real() + ret.imag();
  }
  double ret;
  // specialize x_max == 0
  if(x_mask == 0) {
    this->ExecuteSum(&ret, expval_pauli_Z_func<data_t>(z_mask), iChunk,  1 );
    return ret;
  }

  // Compute the overall phase of the operator.
  // This is (-1j) ** number of Y terms modulo 4
  auto phase = std::complex<data_t>(initial_phase);
  add_y_phase(num_y, phase);
  this->ExecuteSum(&ret, expval_pauli_XYZ_func<data_t>(x_mask, z_mask, x_max, phase), iChunk, 1 );
  return ret;
}



#ifdef AER_THRUST_CUDA

template <typename data_t> __global__
void dev_apply_register_blocked_gates(thrust::complex<data_t>* data,int num_gates,int num_qubits,int num_matrix,uint_t* qubits,BlockedGateParams* params,thrust::complex<double>* matrix)
{
  uint_t i,idx,ii,t,offset;
  uint_t j,laneID,iPair;
  thrust::complex<data_t> q,qp,qt;
  thrust::complex<double> m0,m1;
  data_t qr,qi;
  int nElem;
  thrust::complex<double>* matrix_load;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  laneID = i & 31;

  //index for this thread
  idx = 0;
  ii = i >> num_qubits;
  for(j=0;j<num_qubits;j++){
    offset = (1ull << qubits[j]);
    t = ii & (offset - 1);
    idx += t;
    ii = (ii - t) << 1;

    if(((laneID >> j) & 1) != 0){
      idx += offset;
    }
  }
  idx += ii;

  q = data[idx];

  //prefetch
  if(threadIdx.x < num_matrix)
    m0 = matrix[threadIdx.x];

  for(j=0;j<num_gates;j++){
    iPair = laneID ^ (1ull << params[j].qubit_);

    matrix_load = matrix;
    nElem = 0;

    switch(params[j].gate_){
    case 'x':
      m0 = 0.0;
      m1 = 1.0;
      break;
    case 'y':
      m0 = 0.0;
      if(iPair > laneID)
        m1 = thrust::complex<double>(0.0,-1.0);
      else
        m1 = thrust::complex<double>(0.0,1.0);
      break;
    case 'p':
      nElem = 1;
      matrix += 1;
      m1 = 0.0;
      break;
    case 'd':
      nElem = 2;
      matrix += 2;
      m1 = 0.0;
      break;
    default:
      nElem = 4;
      matrix += 4;
      break;
    }

    if(iPair < laneID){
      matrix_load += (nElem >> 1);
    }
    if(nElem > 0)
      m0 = *(matrix_load);
    if(nElem > 2)
      m1 = *(matrix_load + 1);

    //warp shuffle to get pair amplitude
    qr = __shfl_sync(0xffffffff,q.real(),iPair,32);
    qi = __shfl_sync(0xffffffff,q.imag(),iPair,32);
    qp = thrust::complex<data_t>(qr,qi);
    qt = m0*q + m1* qp;

    if((idx & params[j].mask_) == params[j].mask_){   //handling control bits
      q = qt;
    }
  }

  data[idx] = q;
}


template <typename data_t> __global__
void dev_apply_shared_memory_blocked_gates(thrust::complex<data_t>* data,int num_gates,int num_qubits,uint_t* qubits,BlockedGateParams* params,thrust::complex<double>* matrix)
{
  __shared__ thrust::complex<data_t> buf[1024];
  uint_t i,idx,ii,t,offset;
  uint_t j,laneID,iPair;
  thrust::complex<data_t> q,qp;
  thrust::complex<double> m0,m1;
  data_t qr,qi;
  int nElem;
  thrust::complex<double>* matrix_load;

  i = blockIdx.x * blockDim.x + threadIdx.x;

  laneID = threadIdx.x;

  //index for this thread
  idx = 0;
  ii = i >> num_qubits;
  for(j=0;j<num_qubits;j++){
    offset = (1ull << qubits[j]);
    t = ii & (offset - 1);
    idx += t;
    ii = (ii - t) << 1;

    if(((laneID >> j) & 1) != 0){
      idx += offset;
    }
  }
  idx += ii;

  q = data[idx];

  for(j=0;j<num_gates;j++){
    iPair = laneID ^ (1ull << params[j].qubit_);

    if(params[j].qubit_ < 5){
      //warp shuffle to get pair amplitude
      qr = q.real();
      qi = q.imag();
      qr = __shfl_sync(0xffffffff,qr,iPair & 31,32);
      qi = __shfl_sync(0xffffffff,qi,iPair & 31,32);
      qp = thrust::complex<data_t>(qr,qi);
    }
    else{
      __syncthreads();
      buf[laneID] = q;
      __syncthreads();
      qp = buf[iPair];
    }

    matrix_load = matrix;
    nElem = 0;

    switch(params[j].gate_){
    case 'x':
      m0 = 0.0;
      m1 = 1.0;
      break;
    case 'y':
      m0 = 0.0;
      if(iPair > laneID)
        m1 = thrust::complex<double>(0.0,-1.0);
      else
        m1 = thrust::complex<double>(0.0,1.0);
      break;
    case 'p':
      nElem = 1;
      matrix += 1;
      m1 = 0.0;
      break;
    case 'd':
      nElem = 2;
      matrix += 2;
      m1 = 0.0;
      break;
    default:
      nElem = 4;
      matrix += 4;
      break;
    }

    if(iPair < laneID){
      matrix_load += (nElem >> 1);
    }
    if(nElem > 0)
      m0 = *(matrix_load);
    if(nElem > 2)
      m1 = *(matrix_load + 1);

    if((idx & params[j].mask_) == params[j].mask_){   //handling control bits
      q = m0*q + m1* qp;
    }
  }

  data[idx] = q;
}

#endif

//do all gates stored in queue
template <typename data_t>
void StatevectorChunkContainer<data_t>::apply_blocked_gates(uint_t iChunk)
{
  if(this->num_matrices_ == 1 && iChunk > 1 && iChunk < this->num_chunks_){
    //only the first chunk can apply
    return;
  }
  uint_t iBlock;
  if(iChunk >= this->num_chunks_){  //for buffer chunks
    iBlock = this->num_matrices_ + iChunk - this->num_chunks_;
  }
  else{
    iBlock = iChunk;
  }

  if(this->num_blocked_gates_[iBlock] == 0)
    return;

#ifdef AER_THRUST_CUDA

  uint_t size;
  uint_t* pQubits;
  BlockedGateParams* pParams;
  thrust::complex<double>* pMatrix;

  this->set_device();

  pQubits = this->param_pointer(iChunk);
  pParams = (BlockedGateParams*)(this->param_pointer(iChunk) + this->num_blocked_qubits_[iBlock]);
  pMatrix = this->matrix_pointer(iChunk);

  if(this->num_matrices_ == 1){
    size = this->num_chunks_ << this->chunk_bits_;
  }
  else{
    size = 1ull << this->chunk_bits_;
  }
  uint_t nt,nb;
  nt = size;
  nb = 1;
  if(nt > 1024){
    nb = (nt + 1024 - 1) / 1024;
    nt = 1024;
  }

  if(this->num_blocked_qubits_[iBlock] < 6){
    //using register blocking (<=5 qubits)
    dev_apply_register_blocked_gates<data_t><<<nb,nt,this->num_blocked_matrix_[iChunk]*sizeof(thrust::complex<double>),this->stream(iChunk)>>>(
                                                                          this->chunk_pointer(iChunk),
                                                                          this->num_blocked_gates_[iBlock],
                                                                          this->num_blocked_qubits_[iBlock],
                                                                          this->num_blocked_matrix_[iBlock],
                                                                          pQubits,pParams,pMatrix);
  }
  else{
    //using shared memory blocking (<=10 qubits)
    dev_apply_shared_memory_blocked_gates<data_t><<<nb,nt,1024*sizeof(data_t),this->stream(iChunk)>>>(
                                                                          this->chunk_pointer(iChunk),
                                                                          this->num_blocked_gates_[iBlock],
                                                                          this->num_blocked_qubits_[iBlock],
                                                                          pQubits,pParams,pMatrix);
  }

#endif

  this->num_blocked_gates_[iBlock] = 0;
  this->num_blocked_matrix_[iBlock] = 0;

}

//------------------------------------------------------------------------------
} // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
