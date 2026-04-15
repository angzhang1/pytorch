# Owner(s): ["module: dynamo"]
import unittest
import weakref

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging
from torch._dynamo.exc import FailOnRecompileLimitHit
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.logging_utils import kwargs_to_settings, log_settings


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class RecompileUxTests(torch._dynamo.test_case.TestCase):
    # TODO(whc) dynamo actually recompiles one more time than the cache limit
    cache_limit = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch("recompile_limit", cls.cache_limit)
        )

    def test_drop_cache_on_skip(self):
        def model(x, i):
            return x + i

        attached = False
        triggered = False

        def trigger():
            nonlocal triggered
            triggered = True

        def compiler(gm, input):
            nonlocal attached
            f = gm.forward
            if attached:
                raise AssertionError("Expected not attached")
            # NB: making this a weakref.ref causes the cycle to no
            # longer be promptly GC'ed
            weakref.finalize(f, trigger)
            attached = True
            return f

        x = torch.randn(2)
        for i in range(2):
            opt_model = torch.compile(model, backend=compiler)
            opt_model(x, i)

        self.assertTrue(triggered)

    def test_loop_torture(self):
        def loop_torture(input, iters):
            out = input
            # randint itself causes one graph break
            for _ in range(iters):
                out += input
            return out

        compile_counter = torch._dynamo.testing.CompileCounter()
        for _ in range(10):
            x = torch.randn(3)
            iters = torch.randint(low=0, high=1000, size=())
            opt_loop_torture = torch.compile(loop_torture, backend=compile_counter)
            opt_loop_torture(x, iters)

        # Currently, we recompile each time,
        # We'd probably like to bail out quickly and warn
        # TODO(whc) these checks fail on py37.  Why?
        # self.assertEqual(counters["frames"]["total"], 2 + self.cache_limit)
        # self.assertEqual(counters["frames"]["ok"], 1 + self.cache_limit)

        # compile_counter only sees frames that were fed to the backend compiler,
        # which is a subset of counters["frames"]["ok"] -- probably because
        # counters["frames"]["ok"] includes frames not containing torch ops?
        self.assertEqual(compile_counter.frame_count, self.cache_limit)

    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_input(self):
        def model(input):
            return input + input

        expected_recompiles = 2
        compile_counter = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch("recompile_limit", expected_recompiles):
            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                for _ in range(10):
                    bsz = torch.randint(low=0, high=1000, size=())
                    x = torch.randn((bsz, 3, 4))
                    opt_model = torch.compile(model, backend=compile_counter)
                    opt_model(x)

        self.assertEqual(compile_counter.frame_count, expected_recompiles)
        self.assertEqual(len(logs.records), 1)
        print(logs.records[0])
        self.assertTrue(
            logs.records[0]
            .getMessage()
            .startswith("torch._dynamo hit config.recompile_limit")
        )

    @unittest.skipIf(
        not torch.cuda.is_available() and not torch.xpu.is_available(),
        "requires cuda or xpu",
    )
    def test_nvfuser_guards(self):
        # we may want to model dynamo's guards sufficiently after nvfuser's ProfilingExecutor guards
        # such that we ensure dynamo is in charge of all the recompilations at the top level,
        # and we could thus simplify the underlying torchscript executor
        def func(a, b, c):
            return a + b * c

        a = torch.rand(3, 4, 5, device=device_type)
        b = torch.rand(3, 4, 5, device=device_type)
        b_v = torch.rand(3, 5, 4, device=device_type).view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device=device_type).permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device=device_type)
        compile_counter = torch._dynamo.testing.CompileCounter()

        with torch._dynamo.config.patch("recompile_limit", 2):
            opt_func = torch.compile(func, backend=compile_counter)
            opt_func(a, b, c)  # warmup
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b, c)  # no guard fail or recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_v, c)  # a view should not cause nvfuser recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_p, c)  # a permutation should cause recompile
            self.assertEqual(compile_counter.frame_count, 2)

    def assert_single_log_contains(self, logs, contains_str):
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(
            logs.records[0].getMessage().find(contains_str) > 0,
            msg=f'Expected to find "{contains_str}" in log "{logs.records[0].getMessage()}"',
        )

    def test_verbose_tensor_check(self):
        def func(a):
            # Warning: choose a function here whose meta implementation lives
            # entirely in C++.  If you do a Python one, Dynamo will dive into
            # torch._refs which is OK but it will muddy up the warnings
            return torch.add(a, 4)

        def cache_fail_test(cached_input, missed_input, expected_failure):
            # TODO(whc) maybe its hacky to have a 'test within a test' but this seemed convenient
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            opt_func = torch.compile(func, backend="eager")
            # warmup
            opt_func(cached_input)

            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                opt_func = torch.compile(func, backend="eager")
                opt_func(missed_input)
            self.assert_single_log_contains(logs, expected_failure)

        a = torch.rand(3, 4, 5)
        cache_fail_test(
            a,
            a[0:2, :, :],
            "tensor 'a' size mismatch at index 0. expected 3, actual 2",
        )
        cache_fail_test(
            a,
            a.clone().as_strided((3, 4, 5), stride=(1, 3, 12)),
            "tensor 'a' stride mismatch at index 0. expected 20, actual 1",
        )
        cache_fail_test(a, a[0, :, :], "tensor 'a' rank mismatch. expected 3, actual 2")
        cache_fail_test(a, a.to("meta"), "tensor 'a' dispatch key set mismatch.")
        cache_fail_test(
            a,
            a.to(torch.float16),
            "tensor 'a' dtype mismatch. expected Float, actual Half",
        )
        a_grad = a.clone()
        a_grad.requires_grad = True
        cache_fail_test(
            a,
            a_grad,
            "tensor 'a' requires_grad mismatch. expected requires_grad=0",
        )

    def test_mismatched_type(self):
        a = torch.rand(3, 4, 5)
        b = torch.rand(3, 4, 5)

        def func(a, b):
            return a + b

        opt_func = torch.compile(func, backend="eager")
        # warmup
        opt_func(a, b)

        with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
            opt_func = torch.compile(func, backend="eager")
            opt_func(a, 1)
        self.assert_single_log_contains(
            logs,
            "expected type of 'b' to be a tensor type, ' but found <class 'int'>",
        )

    @torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True)
    def test_fail_on_recompile_limit_hit(self):
        @torch.compile(backend="eager")
        def func(b, a):
            if a:
                return b * 2
            else:
                return b + 1

        func(torch.randn(5), True)
        with self.assertRaises(FailOnRecompileLimitHit):
            func(torch.randn(5), False)

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails(self):
        failure_reasons = []

        def guard_fail_fn(failure):
            failure_reasons.append(failure[0])

        def f(x):
            return torch.relu(x)

        opt_f = torch._dynamo.optimize(
            backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
        )(f)

        for i in range(5):
            failure_reasons.clear()
            opt_f(torch.randn(8 + i))

        failure_str = "\n".join(failure_reasons)
        for line in [
            "tensor 'x' size mismatch at index 0. expected 11, actual 12",
            "tensor 'x' size mismatch at index 0. expected 10, actual 12",
            "tensor 'x' size mismatch at index 0. expected 9, actual 12",
            "tensor 'x' size mismatch at index 0. expected 8, actual 12",
        ]:
            self.assertIn(
                line,
                failure_str,
            )

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails_report_all(self):
        with log_settings(kwargs_to_settings(recompiles_verbose=True)):
            failure_reasons = []

            def guard_fail_fn(failure):
                failure_reasons.append(failure[0])

            def f(x):
                return torch.ones(len(x), x[-1])

            opt_f = torch._dynamo.optimize(
                backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
            )(f)

            opt_f([4, 5, 6])

            def filter_reasons():
                return "\n".join(
                    [
                        line
                        for line in "\n".join(failure_reasons).splitlines()
                        if not line.startswith("___check_type_id")
                    ]
                )

            failure_reasons.clear()
            opt_f([7, 8])

            for line in ["len(x) == 3"]:
                self.assertIn(line, filter_reasons())

            failure_reasons.clear()
            opt_f([9])

            for line in ["len(x) == 2", "len(x) == 3"]:
                self.assertIn(line, filter_reasons())

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_recompile_child_run_only(self):
        def f(x, n):
            if torch.compiler.is_compiling():
                x = x + 1
            x = g(x)
            return h(x) + n

        def g(x):
            if torch.compiler.is_compiling():
                return x + 2
            return x

        def h(x):
            if torch.compiler.is_compiling():
                return x + 4
            return x

        torch.compile(g, backend="eager")(torch.randn(3))
        inp = torch.randn(3)
        opt_f = torch.compile(f, backend="eager")
        opt_f(inp, 0)

        # expect f to run eager, g compiled (from previous invocatino), h eager
        res = opt_f(inp, 1)

        self.assertEqual(res, inp + 3)


class RecompileLimitKwargTests(torch._dynamo.test_case.TestCase):
    @staticmethod
    def _num_cache_entries(code):
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))

    def test_recompile_limit_basic(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        opt_f = torch.compile(f, backend=cnt, recompile_limit=2)

        opt_f(torch.randn(3), torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 1)

        opt_f(torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64))
        self.assertEqual(self._num_cache_entries(f), 2)

        # Third dtype should NOT trigger recompilation (recompile_limit=2)
        opt_f(torch.randn(3, dtype=torch.float16), torch.randn(3, dtype=torch.float16))
        self.assertEqual(self._num_cache_entries(f), 2)

    def test_recompile_limit_none_uses_global(self):
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        # Without recompile_limit kwarg, uses global config (default 8)
        opt_f = torch.compile(f, backend=cnt)

        for i in range(10):
            dtype = [
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
                torch.int32,
                torch.int64,
                torch.int16,
                torch.int8,
                torch.uint8,
                torch.complex64,
            ][i]
            opt_f(torch.ones(3, dtype=dtype), torch.ones(3, dtype=dtype))

        self.assertEqual(
            self._num_cache_entries(f), torch._dynamo.config.recompile_limit
        )

    def test_recompile_limit_fullgraph_raises(self):
        """With fullgraph=True, hitting the recompile_limit kwarg raises
        FailOnRecompileLimitHit, consistent with the fullgraph contract."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_f = torch.compile(f, backend=cnt, fullgraph=True, recompile_limit=1)

        opt_f(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        with self.assertRaises(FailOnRecompileLimitHit):
            opt_f(torch.randn(3, dtype=torch.float64))

    def test_recompile_limit_stricter_than_global(self):
        """recompile_limit kwarg can be stricter than the global config."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        # Global default is 8, but this region only allows 1
        opt_f = torch.compile(f, backend=cnt, recompile_limit=1)

        opt_f(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # Should stop — recompile_limit=1 reached
        opt_f(torch.randn(3, dtype=torch.float64))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_recompile_limit_resume_function_auto_dynamic(self):
        """With automatic dynamic shapes and recompile_limit=2, the resume
        function recompiles via dimension changes on a global tensor while
        the main function gets cache hits. The resume function should stop
        at 2 entries and fall back to eager."""
        cnt = torch._dynamo.testing.CompileCounter()

        y_holder = {"tensor": torch.randn(4, 8, 2)}

        def f(x):
            x.sin()
            print("graph break")
            return y_holder["tensor"].cos()

        opt_f = torch.compile(f, backend=cnt, recompile_limit=2)

        # Call 1: static compile
        y_holder["tensor"] = torch.randn(4, 8, 2)
        opt_f(torch.randn(4, 8, 2))

        # Call 2: y dim0 changes -> f cache hit, resume recompiles
        y_holder["tensor"] = torch.randn(5, 8, 2)
        opt_f(torch.randn(4, 8, 2))
        frame_count_after_2 = cnt.frame_count

        # Call 3: y dim1 changes -> resume should NOT recompile
        # (resume already has 2 entries = recompile_limit)
        y_holder["tensor"] = torch.randn(5, 9, 2)
        opt_f(torch.randn(4, 8, 2))
        self.assertEqual(cnt.frame_count, frame_count_after_2)

        # Verify f has 1 entry, resume has 2
        num_f_entries = len(torch._dynamo.eval_frame._debug_get_cache_entry_list(f))
        self.assertEqual(num_f_entries, 1)

        from torch._dynamo.resume_execution import ContinueExecutionCache

        resume_codes = list(ContinueExecutionCache.cache[f.__code__].values())
        self.assertTrue(len(resume_codes) > 0, "No resume functions found")
        for resume_code in resume_codes:
            num_resume_entries = len(
                torch._dynamo.eval_frame._debug_get_cache_entry_list(resume_code)
            )
            self.assertEqual(num_resume_entries, 2)


class IsolateRecompilesTests(torch._dynamo.test_case.TestCase):
    """Tests for isolate_recompiles=True on torch.compile(). Each compile call
    gets its own isolated cache via the per-compile cache map."""

    @staticmethod
    def _num_cache_entries(code):
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_basic(self):
        """Basic isolation: recompile limit errors within an isolated region."""

        def f(x):
            return x.sin()

        opt_f = torch.compile(
            f, backend="eager", dynamic=False, isolate_recompiles=True
        )

        opt_f(torch.randn(3))

        with self.assertRaises(FailOnRecompileLimitHit):
            opt_f(torch.randn(4))

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_same_function_different_regions(self):
        """Two torch.compile() calls on the same function with isolate_recompiles
        get fully independent caches. Each can compile without the other's
        entries interfering."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))

        # Both compiled independently — no FailOnRecompileLimitHit
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        recompile_limit=1,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_factory_pattern(self):
        """Factory creates multiple torch.compile wrappers around the same
        inner function. Each gets its own isolated cache."""
        from functools import cache

        def core(x):
            return x.sum()

        @cache
        def factory(key):
            @torch.compile(fullgraph=True, dynamic=False, isolate_recompiles=True)
            def frontend(x, n):
                return core(x) + n

            return frontend

        factory("foo")(torch.ones(3), 3)
        factory("bar")(torch.ones(4), 3)
        factory("baz")(torch.ones(5), 3)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_static_and_dynamic(self):
        """Two compile regions on the same function with the same backend:
        one static, one dynamic. isolate_recompiles keeps their cache entries
        separate (without it the static recompile would count against the
        dynamic region too)."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_static = torch.compile(
            f, backend=cnt, dynamic=False, isolate_recompiles=True
        )
        opt_dynamic = torch.compile(
            f, backend=cnt, dynamic=True, isolate_recompiles=True
        )

        opt_static(torch.randn(4, 8))
        self.assertEqual(cnt.frame_count, 1)

        opt_dynamic(torch.randn(5, 9))
        self.assertEqual(cnt.frame_count, 2)

        # Static cache hit — no new compilation
        opt_static(torch.randn(4, 8))
        self.assertEqual(cnt.frame_count, 2)

        # Dynamic cache hit with different shape — no new compilation
        opt_dynamic(torch.randn(6, 10))
        self.assertEqual(cnt.frame_count, 2)

        # Static recompile with new shape
        opt_static(torch.randn(5, 9))
        self.assertEqual(cnt.frame_count, 3)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_no_isolate_recompiles_shared_cache(self):
        """Without isolate_recompiles, two compile calls on the same function
        share the cache — a static recompile from one counts against the
        other's limit."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_a = torch.compile(f, backend=cnt, dynamic=False)
        opt_b = torch.compile(f, backend=cnt, dynamic=False)

        opt_a(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)

        # opt_b shares the cache — same shape is a cache hit
        opt_b(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)

        # Different shape triggers a recompile counted in the shared cache
        opt_a(torch.randn(5))
        self.assertEqual(cnt.frame_count, 2)

        # opt_b sees the shared recompilation
        opt_b(torch.randn(5))
        self.assertEqual(cnt.frame_count, 2)

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_different_backends_shared_cache_without_isolate(self):
        """Without isolate_recompiles, different backends on the same function
        share the cache. Entries from backend A count against backend B's
        recompile limit even though they don't cross-match during lookup."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_a = torch.compile(f, backend=cnt_a, dynamic=False)
        opt_b = torch.compile(f, backend=cnt_b, dynamic=False)

        # Backend A compiles twice (two shapes)
        opt_a(torch.randn(3))
        opt_a(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)

        # Backend B: despite being a different backend, backend A's entries
        # fill the shared cache, pushing B over the recompile limit.
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_b(torch.randn(5))

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_different_backends_independent_with_isolate(self):
        """With isolate_recompiles, different backends on the same function
        get independent cache buckets."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sum()

        opt_a = torch.compile(f, backend=cnt_a, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, dynamic=False, isolate_recompiles=True)

        # Backend A compiles twice
        opt_a(torch.randn(3))
        opt_a(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)

        # Backend B compiles independently — not affected by A's entries
        opt_b(torch.randn(5))
        self.assertEqual(cnt_b.frame_count, 1)
        opt_b(torch.randn(6))
        self.assertEqual(cnt_b.frame_count, 2)

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_isolate_recompiles_fullgraph_raises(self):
        """With fullgraph=True, hitting the recompile limit raises
        FailOnRecompileLimitHit."""

        def f(x):
            return x.sin()

        opt_f = torch.compile(
            f, backend="eager", fullgraph=True, dynamic=False, isolate_recompiles=True
        )

        opt_f(torch.randn(3))
        with self.assertRaisesRegex(FailOnRecompileLimitHit, "fullgraph=True"):
            opt_f(torch.randn(4))

    def test_isolate_recompiles_mark_dynamic_vs_static(self):
        """Two regions on the same function: one with mark_static, one with
        mark_dynamic. Their guards don't interfere."""
        cnt_static = torch._dynamo.testing.CompileCounter()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_static = torch.compile(f, backend=cnt_static, isolate_recompiles=True)
        opt_dynamic = torch.compile(f, backend=cnt_dynamic, isolate_recompiles=True)

        x_static = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static, 0)
        opt_static(x_static)
        self.assertEqual(cnt_static.frame_count, 1)

        x_dynamic = torch.randn(4, 8)
        torch._dynamo.mark_dynamic(x_dynamic, 0)
        opt_dynamic(x_dynamic)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static cache hit — same shape
        x_static2 = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static2, 0)
        opt_static(x_static2)
        self.assertEqual(cnt_static.frame_count, 1)

        # Dynamic cache hit — different shape, same dynamic dim
        x_dynamic2 = torch.randn(7, 8)
        opt_dynamic(x_dynamic2)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static recompile — different shape
        x_static3 = torch.randn(7, 8)
        torch._dynamo.mark_static(x_static3, 0)
        opt_static(x_static3)
        self.assertEqual(cnt_static.frame_count, 2)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_isolate_recompiles_auto_dynamic_shared_pgo(self):
        """With isolate_recompiles, PGO (frame_state) is shared. Region B
        benefits from region A's shape observations."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt_a, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, isolate_recompiles=True)

        opt_a(torch.randn(3, 4))
        opt_a(torch.randn(5, 4))
        self.assertEqual(cnt_a.frame_count, 2)

        # Region B benefits from A's PGO — compiles dynamic immediately
        opt_b(torch.randn(7, 4))
        self.assertEqual(cnt_b.frame_count, 1)

        opt_b(torch.randn(9, 4))
        self.assertEqual(cnt_b.frame_count, 1)

    @torch._dynamo.config.patch(
        accumulated_recompile_limit=3, recompile_limit=8, automatic_dynamic_shapes=False
    )
    def test_isolate_recompiles_accumulated_limit(self):
        """Even with isolated regions, accumulated_recompile_limit applies
        per-region. Once a region hits it, no more compilations."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_f = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)

        opt_f(torch.randn(3))
        opt_f(torch.randn(4))
        opt_f(torch.randn(5))
        self.assertEqual(cnt.frame_count, 3)

        # Accumulated limit hit — no more compilations
        opt_f(torch.randn(6))
        self.assertEqual(cnt.frame_count, 3)

    def test_non_isolated_entries_visible_to_isolated(self):
        """Non-isolated (region -1) cache entries are visible to isolated
        region lookups via read-only fallback, provided the backend matches.
        This is BC friendly — isolated compiles reuse existing work."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.exp()

        opt_global = torch.compile(f, backend=cnt)
        opt_global(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # Isolated region with SAME backend — falls back to global entry
        opt_isolated = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_isolated(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_non_isolated_compiles_share_cache(self):
        """Two non-isolated torch.compile() calls on the same function share
        the same cache bucket (-1). They share cache hits AND share the
        recompile limit — a compilation from either counts toward the same limit."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.exp()

        opt_a = torch.compile(f, backend=cnt, dynamic=False)
        opt_b = torch.compile(f, backend=cnt, dynamic=False)

        # opt_a compiles shape 3 — entry 1 of 2
        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # opt_b gets cache hit on shape 3
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # opt_b compiles shape 4 — entry 2 of 2 (shared limit)
        opt_b(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        # opt_a gets cache hit on shape 4 (from opt_b)
        opt_a(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        # Either hitting shape 5 exceeds the shared recompile_limit
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_a(torch.randn(5))

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_lru_per_region(self):
        """LRU reordering is per-region. Verify by checking that cache entries
        within each region maintain correct ordering after hits."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt_a, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, dynamic=False, isolate_recompiles=True)

        # Populate region A: shape 3 then shape 4
        opt_a(torch.randn(3))
        opt_a(torch.randn(4))

        # Populate region B: shape 5 then shape 6
        opt_b(torch.randn(5))
        opt_b(torch.randn(6))

        self.assertEqual(cnt_a.frame_count, 2)
        self.assertEqual(cnt_b.frame_count, 2)

        # Hit region A shape 3 — moves it to front (LRU)
        opt_a(torch.randn(3))

        # Repeated hits on both — no recompilations
        for _ in range(5):
            opt_a(torch.randn(3))
            opt_a(torch.randn(4))
            opt_b(torch.randn(5))
            opt_b(torch.randn(6))

        self.assertEqual(cnt_a.frame_count, 2)
        self.assertEqual(cnt_b.frame_count, 2)
        self.assertEqual(self._num_cache_entries(f), 4)

    def test_isolate_recompiles_reset(self):
        """torch._dynamo.reset() clears all regions."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.cos()

        opt_a = torch.compile(f, backend=cnt_a, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt_b, isolate_recompiles=True)

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 1)
        self.assertEqual(cnt_b.frame_count, 1)

        torch._dynamo.reset()

        opt_a(torch.randn(3))
        opt_b(torch.randn(4))
        self.assertEqual(cnt_a.frame_count, 2)
        self.assertEqual(cnt_b.frame_count, 2)

    @torch._dynamo.config.patch(recompile_limit=3)
    def test_isolate_recompiles_resume_function(self):
        """Resume functions from a graph break inside an isolated region
        inherit the isolate_recompiles_id and respect the per-compile recompile limit."""
        cnt = torch._dynamo.testing.CompileCounter()

        mode = {"value": "a"}

        def f(x):
            a = x.sin()
            torch._dynamo.graph_break()
            if mode["value"] == "a":
                return a.cos()
            elif mode["value"] == "b":
                return a.tan()
            elif mode["value"] == "c":
                return a.exp()
            else:
                return a + 1

        opt_f = torch.compile(f, backend=cnt, isolate_recompiles=True)

        opt_f(torch.randn(4))
        frame_count_after_1 = cnt.frame_count

        mode["value"] = "b"
        opt_f(torch.randn(4))
        frame_count_after_2 = cnt.frame_count
        self.assertGreater(frame_count_after_2, frame_count_after_1)

        mode["value"] = "c"
        opt_f(torch.randn(4))
        frame_count_after_3 = cnt.frame_count
        self.assertGreater(frame_count_after_3, frame_count_after_2)

        # Resume function has 3 entries = recompile_limit. Fourth blocked.
        mode["value"] = "d"
        opt_f(torch.randn(4))
        self.assertEqual(cnt.frame_count, frame_count_after_3)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=False)
    def test_isolate_recompiles_same_backend_different_regions(self):
        """Two isolated regions using the SAME CompileCounter backend.
        Without proper C++ cache bucketing, the second region would get a
        cache hit from the first region's entry (same backend, same guards).
        This verifies the per-region map is actually used."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=cnt, dynamic=False, isolate_recompiles=True)

        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # Must compile again — different region, even though same backend + input
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)

        # Cache hits within each region
        opt_a(torch.randn(3))
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)

    @parametrize("backend", ["eager", "aot_eager", "inductor"])
    def test_isolate_recompiles_string_backends(self, backend):
        """Two isolated regions using the same string backend compile
        independently — verified by cache entry count."""

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=backend, isolate_recompiles=True)
        opt_b = torch.compile(f, backend=backend, isolate_recompiles=True)

        opt_a(torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 1)

        opt_b(torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 2)

        # Cache hits
        opt_a(torch.randn(3))
        opt_b(torch.randn(3))
        self.assertEqual(self._num_cache_entries(f), 2)

    def test_isolate_recompiles_gc_wrapper(self):
        """When an isolated region's compile wrapper is GC'd, orphaned cache
        entries remain but a new torch.compile gets a fresh isolate_recompiles_id."""
        import gc

        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_a(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(self._num_cache_entries(f), 1)

        # Drop the wrapper and force GC
        del opt_a
        gc.collect()

        # Orphaned entry still on the code object
        self.assertEqual(self._num_cache_entries(f), 1)

        # New compile gets a fresh region — compiles independently,
        # doesn't reuse the orphaned entry
        opt_b = torch.compile(f, backend=cnt, isolate_recompiles=True)
        opt_b(torch.randn(3))
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(self._num_cache_entries(f), 2)

        # reset() clears everything including orphaned entries
        torch._dynamo.reset()
        self.assertEqual(self._num_cache_entries(f), 0)

    @torch._dynamo.config.patch(
        recompile_limit=2,
        fail_on_recompile_limit_hit=True,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_same_backend_different_dynamic_independent_limits(self):
        """Two isolated regions on the same function and same backend, one
        static and one dynamic. Each region should exhaust its recompile_limit
        independently without the other's entries counting against it."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_static = torch.compile(
            f, backend=cnt, dynamic=False, isolate_recompiles=True
        )
        opt_dynamic = torch.compile(
            f, backend=cnt, dynamic=True, isolate_recompiles=True
        )

        # Region A (static): two shapes fill up recompile_limit=2
        opt_static(torch.randn(3))
        opt_static(torch.randn(4))
        self.assertEqual(cnt.frame_count, 2)

        # Region B (dynamic): compiles once, different shapes are cache hits
        opt_dynamic(torch.randn(5))
        self.assertEqual(cnt.frame_count, 3)
        opt_dynamic(torch.randn(6))
        self.assertEqual(cnt.frame_count, 3)

        # Region A hits recompile_limit — third shape raises
        with self.assertRaises(FailOnRecompileLimitHit):
            opt_static(torch.randn(5))

        # Region B still works — its limit is independent
        opt_dynamic(torch.randn(7))
        self.assertEqual(cnt.frame_count, 3)

    @torch._dynamo.config.patch(
        recompile_limit=1,
        automatic_dynamic_shapes=False,
    )
    def test_isolate_recompiles_limit_does_not_skip_other_regions(self):
        """When an isolated region hits its recompile limit and falls back
        to eager, the default (non-isolated) region can still compile."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_isolated = torch.compile(
            f, backend=cnt, dynamic=False, isolate_recompiles=True
        )
        opt_default = torch.compile(f, backend=cnt, dynamic=False)

        # Isolated region compiles once
        opt_isolated(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        # Isolated region hits limit — falls back to eager silently
        opt_isolated(torch.randn(4))
        self.assertEqual(cnt.frame_count, 1)

        # Default region still compiles — not affected by isolated limit
        opt_default(torch.randn(5))
        self.assertEqual(cnt.frame_count, 2)

    def test_debug_get_cache_entry_list_deterministic_order(self):
        """_debug_get_cache_entry_list returns entries sorted by
        isolate_recompiles_id, so the order is deterministic regardless
        of unordered_map iteration."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        # Create entries across several isolated regions in reverse order
        # to make unordered_map iteration more likely to differ from sorted.
        opts = [
            torch.compile(f, backend=cnt, isolate_recompiles=True) for _ in range(6)
        ]
        for opt in reversed(opts):
            opt(torch.randn(3))

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(f)
        self.assertEqual(len(entries), 6)

        ids = [e.isolate_recompiles_id for e in entries]
        self.assertEqual(ids, sorted(ids))


instantiate_parametrized_tests(IsolateRecompilesTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
