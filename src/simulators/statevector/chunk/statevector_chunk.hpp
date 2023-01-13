/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _qv_statevector_chunk_hpp_
#define _qv_statevector_chunk_hpp_

#include "simulators/statevector/chunk/device_chunk_container.hpp"
#include "simulators/statevector/chunk/host_chunk_container.hpp"

#include "simulators/statevector/chunk/chunk.hpp"

#ifdef AER_CUSTATEVEC
#include "simulators/statevector/chunk/cuStateVec_chunk_container.hpp"
#endif


namespace AER {
namespace QV {
namespace Chunk {


//============================================================================
// chunk class
//============================================================================
template <typename data_t>
class StatevectorChunk : public Chunk<thrust::complex<data_t>>
{
protected:
  std::shared_ptr<StatevectorChunk<data_t>> cache_;                 //pointer to cache chunk on device
public:
  StatevectorChunk()
  {
    cache_ = nullptr;
  }

  ~StatevectorChunk()
  {
    if(cache_)
      cache_.reset();
  }

  void map_cache(Chunk<data_t>& chunk)
  {
    cache_ = std::make_shared<Chunk<data_t>>(chunk);
  }
  void unmap_cache(void)
  {
    if(cache_){
      cache_->unmap();
      cache_.reset();
      cache_ = nullptr;
    }
  }

  void unmap(void) override
  {
    if(cache_)
      unmap_cache();
    Chunk<thrust::complex<data_t>>::unmap();
  }

  void StoreMatrix(const std::vector<std::complex<double>>& mat)
  {
    if(cache_){
      cache_->StoreMatrix(mat);
    }
    else{
      chunk_container_.lock()->StoreMatrix(mat,chunk_pos_);
    }
  }
  void StoreMatrix(const std::complex<double>* mat,uint_t size)
  {
    if(cache_){
      cache_->StoreMatrix(mat,size);
    }
    else{
      chunk_container_.lock()->StoreMatrix(mat,chunk_pos_,size);
    }
  }
  void StoreUintParams(const std::vector<uint_t>& prm)
  {
    if(cache_){
      cache_->StoreUintParams(prm);
    }
    else{
      chunk_container_.lock()->StoreUintParams(prm,chunk_pos_);
    }
  }

  void ResizeMatrixBuffers(int bits)
  {
    //synchronize all kernel execution before changing matrix buffer size
    chunk_container_.lock()->synchronize(chunk_pos_);
    chunk_container_.lock()->ResizeMatrixBuffers(bits);
  }



  virtual reg_t sample_measure(const std::vector<double> &rnds,uint_t stride = 1,bool dot = true,uint_t count = 1) const
  {
    return chunk_container_.lock()->sample_measure(chunk_pos_,rnds,stride,dot,count);
  }

  virtual double norm(uint_t count) const
  {
    return chunk_container_.lock()->norm(chunk_pos_,count);
  }
  virtual double trace(uint_t row, uint_t count) const
  {
    return chunk_container_.lock()->trace(chunk_pos_,row,count);
  }

  //apply matrix
  virtual void apply_matrix(const reg_t& qubits,const int_t control_bits,const cvector_t<double> &mat,const uint_t count)
  {
    if(cache_)
      cache_->chunk_container_.lock()->apply_matrix(cache_->chunk_pos_, qubits,control_bits,mat,chunk_index_,count);
    else
      chunk_container_.lock()->apply_matrix(chunk_pos_,qubits,control_bits,mat,chunk_index_,count);
  }
  //apply diagonal matrix
  virtual void apply_diagonal_matrix(const reg_t& qubits,const int_t control_bits,const cvector_t<double> &diag,const uint_t count)
  {
    if(cache_)
      cache_->chunk_container_.lock()->apply_diagonal_matrix(cache_->chunk_pos_, qubits,control_bits,diag,chunk_index_,count);
    else
      chunk_container_.lock()->apply_diagonal_matrix(chunk_pos_,qubits,control_bits,diag,chunk_index_,count);
  }
  //apply (controlled) X
  virtual void apply_X(const reg_t& qubits,const uint_t count)
  {
    if(cache_)
      cache_->chunk_container_.lock()->apply_X(cache_->chunk_pos_, qubits,chunk_index_,count);
    else
      chunk_container_.lock()->apply_X(chunk_pos_,qubits,chunk_index_,count);
  }
  //apply (controlled) Y
  virtual void apply_Y(const reg_t& qubits,const uint_t count)
  {
    if(cache_)
      cache_->chunk_container_.lock()->apply_Y(cache_->chunk_pos_, qubits,chunk_index_,count);
    else
      chunk_container_.lock()->apply_Y(chunk_pos_,qubits,chunk_index_,count);
  }
  //apply (controlled) phase
  virtual void apply_phase(const reg_t& qubits,const int_t control_bits,const std::complex<double> phase,const uint_t count)
  {
    if(cache_)
      cache_->chunk_container_.lock()->apply_phase(cache_->chunk_pos_, qubits,control_bits,phase,chunk_index_,count);
    else
      chunk_container_.lock()->apply_phase(chunk_pos_,qubits,control_bits,phase,chunk_index_,count);
  }
  //apply (controlled) swap gate
  virtual void apply_swap(const reg_t& qubits,const int_t control_bits,const uint_t count)
  {
    if(cache_)
      cache_->chunk_container_.lock()->apply_swap(cache_->chunk_pos_, qubits,control_bits,chunk_index_,count);
    else
      chunk_container_.lock()->apply_swap(chunk_pos_,qubits,control_bits,chunk_index_,count);
  }
  //apply multiple swap gates
  virtual void apply_multi_swaps(const reg_t& qubits,const uint_t count)
  {
    if(cache_)
      cache_->chunk_container_.lock()->apply_multi_swaps(cache_->chunk_pos_, qubits,chunk_index_,count);
    else
      chunk_container_.lock()->apply_multi_swaps(chunk_pos_,qubits,chunk_index_,count);
  }
  //apply permutation
  virtual void apply_permutation(const reg_t& qubits,const std::vector<std::pair<uint_t, uint_t>> &pairs, const uint_t count)
  {
    if(cache_)
      cache_->chunk_container_.lock()->apply_permutation(cache_->chunk_pos_, qubits,pairs,chunk_index_,count);
    else
      chunk_container_.lock()->apply_permutation(chunk_pos_,qubits,pairs,chunk_index_,count);
  }

  //apply rotation around axis
  virtual void apply_rotation(const reg_t &qubits, const Rotation r, const double theta, const uint_t count)
  {
    if(cache_)
      cache_->chunk_container_.lock()->apply_rotation(cache_->chunk_pos_, qubits,r,theta,chunk_index_,count);
    else
      chunk_container_.lock()->apply_rotation(chunk_pos_,qubits,r,theta,chunk_index_,count);
  }

  //get probabilities of chunk
  virtual void probabilities(std::vector<double>& probs, const reg_t& qubits) const
  {
    chunk_container_.lock()->probabilities(probs, chunk_pos_,qubits);
  }
  //Pauli expectation values
  virtual double expval_pauli(const reg_t& qubits,const std::string &pauli,const complex_t initial_phase) const
  {
    return chunk_container_.lock()->expval_pauli(chunk_pos_,qubits,pauli,initial_phase);
  }


};

//------------------------------------------------------------------------------
}  // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
