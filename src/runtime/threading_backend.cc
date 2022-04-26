/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file threading_backend.cc
 * \brief Native threading backend
 */
#include <tvm/runtime/logging.h>
#include <tvm/runtime/threading_backend.h>

#include <thread>

namespace tvm {
namespace runtime {
namespace threading {
thread_local int max_concurrency = 0;

ThreadGroup::ThreadGroup(int num_workers, std::function<void(int)> worker_callback,
                         bool exclude_worker0)
    : impl_(CreateThreadGroupImpl(num_workers, worker_callback, exclude_worker0)) {}
ThreadGroup::~ThreadGroup() { delete impl_; }
void ThreadGroup::Join() { impl_->Join(); }

int ThreadGroup::Configure(AffinityMode mode, int nthreads, bool exclude_worker0,
                           std::vector<unsigned int> cpus) {
  return impl_->Configure(mode, nthreads, exclude_worker0, cpus);
}

/*!
 * \brief Set the maximum number of available cores.
 */
void SetMaxConcurrency(int value) {
  if (value < 0) {
    LOG(WARNING) << "The value of maximum concurrency '" << value << "' can not be negative "
                 << "the setting of maximum concurrency is not success.";
    return;
  }
  max_concurrency = value;
}
int MaxConcurrency() {
  int max_concurrency = 1;
  if (tvm::runtime::threading::max_concurrency != 0) {
    max_concurrency = tvm::runtime::threading::max_concurrency;
  } else {
    const char* val = getenv("TVM_NUM_THREADS");
    if (val == nullptr) {
      val = getenv("OMP_NUM_THREADS");
    }
    if (val != nullptr) {
      max_concurrency = atoi(val);
    } else {
      max_concurrency = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
      max_concurrency /= 2;  // ignore hyper-threading
#elif defined(__hexagone__)
      // With unsigned PDs, getting the number of available hardware threads
      // is not supported in earlier versions of QuRT. In such cases assume 4.
      // If running on simulator, set max_concurrency to 1.
      if (max_concurrency == 0) {
        if (dlsym(RTLD_DEFAULT, "running_in_sim_dev_17bc90206f6cf5a7")) {
          max_concurrency = 1;
        } else {
          max_concurrency = 4;
        }
      }
#endif
    }
  }
  return std::max(max_concurrency, 1);
}

}  // namespace threading
}  // namespace runtime
}  // namespace tvm
