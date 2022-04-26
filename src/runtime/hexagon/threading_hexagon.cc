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

#include <dlfcn.h>
#include <qurt.h>
#include <stdlib.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/threading_backend.h>

#include <thread>

#define HEXAGON_STACK_SIZE 65536
#define HEXAGON_STACK_ALIGNMENT 32

namespace tvm {
namespace runtime {
namespace threading {

// pthreads are broken on older versions of qurt, so
// we need to use native APIs instead of std::threads
class QuRTThread {
  typedef std::function<void()> Callback;

 public:
  explicit QuRTThread(Callback worker_callback) : worker_callback_(worker_callback) {
    static int id = 1;
    qurt_thread_attr_t attr;
    char name[32];
    int ret = posix_memalign(&stack_, HEXAGON_STACK_ALIGNMENT, HEXAGON_STACK_SIZE);
    CHECK_EQ(ret, 0);
    // When a std::function<> is cast to bool,
    // it indicates whether it stores a callable target
    CHECK_EQ((bool)worker_callback_, true);
    qurt_thread_attr_init(&attr);
    qurt_thread_attr_set_stack_size(&attr, HEXAGON_STACK_SIZE);
    qurt_thread_attr_set_stack_addr(&attr, stack_);
    snprintf(name, sizeof(name), "worker %d", id++);
    qurt_thread_attr_set_name(&attr, name);
    ret = qurt_thread_create(&thread_, &attr, (void (*)(void*))RunFunction, this);
    CHECK_EQ(ret, QURT_EOK);
  }
  QuRTThread(QuRTThread&& other)
      : thread_(other.thread_),
        worker_callback_(std::move(other.worker_callback_)),
        stack_(other.stack_) {
    other.thread_ = 0;
    other.stack_ = nullptr;
  }
  ~QuRTThread() {
    if (thread_) {
      join();
    }
    if (stack_) {
      free(stack_);
    }
  }
  bool joinable() const { return qurt_thread_get_id() != thread_; }
  void join() {
    int status;
    qurt_thread_join(thread_, &status);
  }

 private:
  static void RunFunction(QuRTThread* qrt_thread) {
    qrt_thread->worker_callback_();
    qurt_thread_exit(QURT_EOK);
  }
  qurt_thread_t thread_;
  Callback worker_callback_;
  void* stack_ = nullptr;
};

class ThreadGroupHexagonImpl : public ThreadGroupImplTemplate<QuRTThread> {
 public:
  ThreadGroupHexagonImpl(int num_workers, std::function<void(int)> worker_callback,
                         bool exclude_worker0)
      : ThreadGroupImplTemplate<QuRTThread>(num_workers, worker_callback, exclude_worker0) {}

  virtual int Configure(ThreadGroup::AffinityMode mode, int nthreads, bool exclude_worker0,
                        std::vector<unsigned int> cpus) {
    int num_workers_used = 0;
    switch (mode) {
      case ThreadGroup::kSpecifyOneCorePerThread:
      case ThreadGroup::kSpecifyThreadShareAllCore:
        num_workers_used = cpus.size();
        break;
      default:
        // use default
        num_workers_used = threading::MaxConcurrency();
    }
    // if a specific number was given, use that
    if (nthreads) {
      num_workers_used = nthreads;
    }
    // if MaxConcurrency restricted the number of workers (e.g., due to
    // hyperthreading), respect the restriction. On CPUs with N logical cores
    // and N/2 physical cores this will set affinity to the first N/2 logical
    // ones.
    num_workers_used = std::min(num_workers_, num_workers_used);
    return num_workers_used;
  }
};

void Yield() { qurt_sleep(1); }

ThreadGroup::Impl* CreateThreadGroupImpl(int num_workers, std::function<void(int)> worker_callback,
                                         bool exclude_worker0) {
  return new ThreadGroupHexagonImpl(num_workers, worker_callback, exclude_worker0);
}

}  // namespace threading
}  // namespace runtime
}  // namespace tvm
