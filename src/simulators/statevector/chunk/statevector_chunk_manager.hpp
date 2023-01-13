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


#ifndef _qv_statevector_chunk_manager_hpp_
#define _qv_statevector_chunk_manager_hpp_

#include "simulators/statevector/chunk/chunk_manager.hpp"
#include "simulators/statevector/chunk/statevector_chunk_container.hpp"


namespace AER {
namespace QV {
namespace Chunk {


//============================================================================
// chunk manager class
//============================================================================
template <typename data_t>
class StatevectorChunkManager : public ChunkManager<thrust::complex<data_t>>
{
protected:
public:
  StatevectorChunkManager(){}

  ~StatevectorChunkManager()
  {
    ChunkManager<thrust::complex<data_t>>::Free();
  }

  std::shared_ptr<ChunkContainer<thrust::complex<data_t>>> new_container(bool onGPU = false) override
  {
    if(onGPU)
      return std::make_shared<StatevectorChunkContainer<data_t>>();
    else
      return std::make_shared<HostChunkContainer<thrust::complex<data_t>>>();
  }
};

//------------------------------------------------------------------------------
} //end namespace Chunk
} // end namespace QV
} // end namespace AER
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
