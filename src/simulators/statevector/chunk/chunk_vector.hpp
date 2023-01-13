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


#ifndef _qv_chunk_container_hpp_
#define _qv_chunk_container_hpp_

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#ifdef AER_THRUST_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
DISABLE_WARNING_POP

#include "misc/wrap_thrust.hpp"

#include "framework/utils.hpp"



namespace AER {
namespace QV {
namespace Chunk {

#ifdef AER_THRUST_CUDA
#define AERDeviceVector thrust::device_vector
#else
#define AERDeviceVector thrust::host_vector
#endif
#define AERHostVector thrust::host_vector

//==================================================================
//  base class for vector storage for chunk
//==================================================================

template <typename data_t>
class BaseVector : public std::enable_shared_from_this<VectorBase<data_t>>
{
protected:
public:
  BaseVector(){}

  virtual ~BaseVector(){}

  virtual void resize(uint_t size) = 0;
  virtual uint_t size(void) = 0;

};


template <typename data_t>
class DeviceVector : public BaseVector<data_t>
{
protected:
  AERDeviceVector<data_t> data_;
public:
  DeviceVector(){}
  DeviceVector(uint_t size) : data_(size);

  ~DeviceVector()
  {
    data_.clear();
    data_.shrink_to_fit();
  }

};

//------------------------------------------------------------------------------
} // end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
