// CPU-only worker-pool lifecycle/publication stress. No GPU/model initialization or tensor computation.
// Exercises both legacy and opt-in host waits using the real pool and public handoff methods.
#include "common.hpp"
#include "cpu_moe.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace {
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}
struct SavedEnv {
    const char* name; bool present; std::string value;
    explicit SavedEnv(const char* n) : name(n), present(std::getenv(n) != nullptr), value(present ? std::getenv(n) : "") {}
    ~SavedEnv() { set_env(name, present ? value.c_str() : nullptr); }
};

uint64_t payload(int job, int tid, int rounds) {
    uint64_t value = 0x9e3779b97f4a7c15ull ^ ((uint64_t)job << 16) ^ (unsigned)tid;
    for (int i = 0; i < rounds; ++i) {
        value ^= value >> 12; value ^= value << 25; value ^= value >> 27;
        value *= 0x2545f4914f6cdd1dull;
    }
    return value;
}

void stress(const char* completion, const char* prompt_completion, int workers, uint64_t& jobs) {
    set_env("STRATA_CPU_COMPLETION_SPINS", completion);
    // Workers really sleep between idle dispatches; this tests generation wakes independently of host waits.
    set_env("STRATA_CPU_WORKER_SPINS", "0");
    auto mailbox = std::make_unique<sq::Mailbox>();
    auto result = std::make_unique<sq::Result>();
    auto pool = std::make_unique<sq::CpuMoe>();
    std::array<uint64_t, 6> output{};
    std::array<int, 6> calls{};
    uint32_t sequence = 1;
    for (int lifecycle = 0; lifecycle < 3; ++lifecycle) {
        set_env("STRATA_PROMPT_COMPLETION_SPINS", prompt_completion);
        pool->start(workers, 1, {sq::ExpertLayerDesc{}}, mailbox.get(), result.get(), false);
        // This deliberately invalid runtime value must never be parsed by a waiter or worker. The
        // explicit startup budget remains immutable until stop joins every worker, then restart reloads.
        set_env("STRATA_PROMPT_COMPLETION_SPINS", "not-a-runtime-setting");
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // all-idle wake after startup
        for (int job = 0; job < 240; ++job) {
            const int slow = job % workers;
            const int mode = job % 8;
            output.fill(0); calls.fill(0);
            // Each callback owns its array element. The host deliberately reads non-atomic outputs only
            // after completion, checking the wait's acquire publication and exactly-once job ownership.
            pool->parallel_for_workers([&, job, slow, mode](int tid) {
                if (mode == 0 && tid == slow) std::this_thread::sleep_for(std::chrono::microseconds(100));
                if (mode == 1 && tid != slow) std::this_thread::yield();
                const int rounds = mode == 2 ? (tid == slow ? 2000 : 0) : mode == 3 ? 64 : 0;
                output[tid] = payload(job, tid, rounds);
                ++calls[tid];
            });
            for (int tid = 0; tid < workers; ++tid) {
                const int rounds = mode == 2 ? (tid == slow ? 2000 : 0) : mode == 3 ? 64 : 0;
                SQ_CHECK(calls[tid] == 1 && output[tid] == payload(job, tid, rounds),
                         "completion publication/ownership failed setting%s workers%d lifecycle%d job%d tid%d",
                         completion ? completion : "unset", workers, lifecycle, job, tid);
            }
            ++jobs;
            if (job % 4 == 0) {
                // Empty-expert layer jobs are a real, fast production case; completion may happen before
                // end_layer/end_step reaches its first load or concurrently with entry into atomic wait.
                mailbox->n_tok = 1 + (job / 4) % sq::kMaxT;
                mailbox->n_miss = 0;
                result->done_seq = 0x7f7f7f7fu;
                pool->begin_layer(0);
                const bool prompt = (job / 4) % 3 != 0;
                if (job % 8 == 0) std::this_thread::yield();
                if (job % 12 == 0) std::this_thread::sleep_for(std::chrono::microseconds(10));
                // Alternate prompt/plain without publishing a mutable mode to workers. Some jobs finish
                // before the first wait load; others race sleep entry. Global legacy + prompt0/64 would
                // deadlock here if the final notification still depended only on the global setting.
                pool->end_layer(prompt);
                SQ_CHECK(result->done_seq == 0x7f7f7f7fu, "empty layer unexpectedly changed output sentinel");
                ++jobs;
                mailbox->ready_seq = sequence++;
                pool->begin_step(mailbox->ready_seq, 0, 1);
                if (job % 8 == 4) std::this_thread::yield();
                SQ_CHECK(pool->end_step(), "empty published step timed out");
                ++jobs;
            }
            if (job % 60 == 59) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Shutdown after completed work must wake idle workers, and the same pool must restart without
        // inheriting a stale job count, generation or callback. Concurrent host schedulers are unsupported.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        pool->stop();
        pool->stop();
    }
    sq::log("CPU completion wait lifecycle/publication PASS: setting=%s prompt=%s workers=%d",
            completion ? completion : "unset", prompt_completion ? prompt_completion : "inherit", workers);
}
} // namespace

int main() {
    SavedEnv completion("STRATA_CPU_COMPLETION_SPINS"), prompt("STRATA_PROMPT_COMPLETION_SPINS"),
             worker("STRATA_CPU_WORKER_SPINS"), profile("STRATA_CPU_PHASE_PROFILE"), cache("STRATA_CPU_IQ3_NIBBLE");
    set_env("STRATA_CPU_PHASE_PROFILE", "0"); set_env("STRATA_CPU_IQ3_NIBBLE", "0");
    std::mutex mutex;
    std::condition_variable changed;
    bool finished = false;
    const auto started = std::chrono::steady_clock::now();
    auto elapsed_seconds = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    };
    std::thread watchdog([&] {
        std::unique_lock<std::mutex> lock(mutex);
        // The matrix doubled from8 to16 combinations; retain every job/assertion and bound the entire
        // suite independently. Per-combination timestamps identify the setting active during a hang.
        if (!changed.wait_for(lock, std::chrono::seconds(120), [&] { return finished; })) {
            std::fprintf(stderr, "CPU completion wait stress exceeded global120s deadline at %.3fs (possible lost wake/deadlock)\n",
                         elapsed_seconds());
            std::fflush(stderr);
            std::_Exit(3);
        }
    });
    uint64_t jobs = 0;
    struct Settings { const char* completion; const char* prompt; };
    const Settings settings[] = {{nullptr,nullptr}, {"-9",nullptr}, {"0",nullptr}, {"64",nullptr},
                                 {nullptr,"0"}, {"-9","64"}, {"0","64"}, {"64","0"}};
    int combination = 0;
    for (const auto& setting : settings) {
        for (int workers : {1, 6}) {
            ++combination;
            sq::log("CPU completion progress START %d/16 elapsed=%.3fs setting=%s prompt=%s workers=%d",
                    combination, elapsed_seconds(), setting.completion ? setting.completion : "unset",
                    setting.prompt ? setting.prompt : "inherit", workers);
            std::fflush(stderr);
            stress(setting.completion, setting.prompt, workers, jobs);
            sq::log("CPU completion progress DONE %d/16 elapsed=%.3fs completed_jobs=%llu",
                    combination, elapsed_seconds(), (unsigned long long)jobs);
            std::fflush(stderr);
        }
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        finished = true;
    }
    changed.notify_one();
    watchdog.join();
    sq::log("CPU COMPLETION WAIT PASS: %llu jobs across16 settings/worker combinations and48 start/stop lifecycles",
            (unsigned long long)jobs);
    return 0;
}
