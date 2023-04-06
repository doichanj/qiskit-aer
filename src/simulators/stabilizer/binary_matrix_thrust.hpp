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


#ifndef _binary_matrix_thrust_hpp_
#define _binary_matrix_thrust_hpp_

#include "misc/warnings.hpp"
DISABLE_WARNING_PUSH
#ifdef AER_THRUST_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
DISABLE_WARNING_POP

#include "misc/wrap_thrust.hpp"

#include "framework/utils.hpp"

#include "binary_vector_thrust.hpp"

namespace AER {
namespace BV {

/*******************************************************************************
 *
 * BinaryMatrix Class for Thrust
 *
  * 2-diemnsion bit array (num_qubits * num_qubits)
 ******************************************************************************/

class BinaryMatrixThrust {
public:
  const static size_t BLOCK_SIZE = 64;
  const static size_t BLOCK_BITS = 6;
  const static size_t BLOCK_MASK = (1ull << BLOCK_BITS) - 1;

  BinaryMatrixThrust(){};

//  explicit BinaryMatrixThrust(uint_t length)
//      : m_length(length), m_block_length((length + BLOCK_SIZE - 1) >> BLOCK_BITS), m_data(length*(length + BLOCK_SIZE - 1) >> BLOCK_BITS, ZERO_){};

//  BinaryMatrixThrust(std::vector<uint_t> mdata)
//      : m_length(mdata.size() << BLOCK_BITS), m_block_length(mdata.size()), m_data(mdata){};

  explicit BinaryMatrixThrust(std::string);

  ~BinaryMatrixThrust();

  void setLength(uint_t length);

  void setValue(bool value, uint_t pos, uint_t row = 0);
  void fillValue(bool value);

  void set0(uint_t pos, uint_t row = 0) { setValue(ZERO_, pos, row); };
  void set1(uint_t pos, uint_t row = 0) { setValue(ONE_, pos, row); };

  bool operator[](const uint_t pos) const;

  bool getValue(const uint_t pos, const uint_t row) const;

  void copy_vector(const BinaryVector& vec, uint_t row);
  void copy_matrix(const BinaryMatrixThrust& mat);
  void copy_out_vector(BinaryVector& vec, uint_t row);

  uint_t getLength() const { return m_length; };

  void makeZero()
  {
    fillValue(0);
  }

  size_t blockSize(void)
  {
    return BLOCK_SIZE;
  }
  size_t blockLength(void) const
  {
    return m_block_length;
  }
  uint_t& operator()(const uint_t pos)
  {
    return raw_reference_cast(m_data[pos]);
  }
  uint_t& operator()(const uint_t pos, const uint_t row)
  {
    return raw_reference_cast(m_data[row*m_block_length + pos]);
  }

  uint_t* pointer(uint_t row) const
  {
    return ((uint_t*)thrust::raw_pointer_cast(m_data.data()) + row*m_block_length);
  }

protected:
  uint_t m_length;
  uint_t m_block_length;
  AERDeviceVector<uint_t> m_data;
  static const uint_t ZERO_;
  static const uint_t ONE_;
};


/*******************************************************************************
 *
 * BinaryVector Class Methods
 *
 ******************************************************************************/

const uint_t BinaryMatrixThrust::ZERO_ = 0ULL;
const uint_t BinaryMatrixThrust::ONE_ = 1ULL;

BinaryMatrixThrust::BinaryMatrixThrust(std::string val) 
{
}

BinaryMatrixThrust::~BinaryMatrixThrust()
{
  m_data.clear();
  m_data.shrink_to_fit();
}

void BinaryMatrixThrust::setLength(uint_t length) 
{
  m_length = length;
  if(m_length == 0)
    m_length = 1;
  m_block_length = (m_length + BLOCK_SIZE - 1) >> BLOCK_BITS;
  m_data.resize(m_block_length*m_length);
//  m_data.assign(m_block_length*length, 0ull);
}


void BinaryMatrixThrust::setValue(bool value, uint_t pos, uint_t row)
{
  auto q = pos >> BLOCK_BITS;
  auto r = pos & BLOCK_MASK;
  if (value)
    m_data[row*m_block_length + q] |= (ONE_ << r);
  else
    m_data[row*m_block_length + q] &= ~(ONE_ << r);
}

void BinaryMatrixThrust::fillValue(bool value)
{
  uint_t val = 0ull - (uint_t)value;

  thrust::fill(thrust::device,m_data.begin(),m_data.end(),val);
}

bool BinaryMatrixThrust::operator[](const uint_t pos) const 
{
  auto q = pos >> BLOCK_BITS;
  auto r = pos & BLOCK_MASK;
  return ((m_data[q] & (ONE_ << r)) != 0);
}

bool BinaryMatrixThrust::getValue(const uint_t pos,const uint_t row) const 
{
  auto q = pos >> BLOCK_BITS;
  auto r = pos & BLOCK_MASK;
  return ((m_data[row*m_block_length + q] & (ONE_ << r)) != 0);
}


void BinaryMatrixThrust::copy_vector(const BinaryVector& vec, uint_t row)
{
  cudaMemcpy(pointer(row), (void*)vec.getData().data(), m_block_length*sizeof(uint_t), cudaMemcpyHostToDevice);

  //  thrust::copy_n(vec.getData().begin(), m_block_length, m_data.begin() + row*m_block_length);
}

void BinaryMatrixThrust::copy_matrix(const BinaryMatrixThrust& mat)
{
  cudaMemcpy(pointer(0), (void*)mat.pointer(0), m_length*m_block_length*sizeof(uint_t), cudaMemcpyDeviceToDevice);
}

void BinaryMatrixThrust::copy_out_vector(BinaryVector& vec, uint_t row)
{
  cudaMemcpy((void*)vec.getData().data(), pointer(row), m_block_length*sizeof(uint_t), cudaMemcpyDeviceToHost);
//  thrust::copy_n(m_data.begin() + row*m_block_length, m_block_length, vec.getData().data());
}

//------------------------------------------------------------------------------
} // end namespace BV
} // AER
//------------------------------------------------------------------------------
#endif