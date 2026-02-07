
import pytest
import torch
from logic.src.envs.jssp import JSSPEnv

class TestJSSPEnv:
    @pytest.fixture
    def env(self):
        # 3 Jobs, 3 Machines
        return JSSPEnv(num_jobs=3, num_machines=3, device="cpu")

    def test_generator(self, env):
        td = env.reset(batch_size=[2])
        assert td["proc_time"].shape == (2, 3, 3)
        assert td["machine_order"].shape == (2, 3, 3)
        assert td["next_op_idx"].shape == (2, 3)
        assert td["finished_jobs"].shape == (2, 3)

    def test_step_logic(self, env):
        td = env.reset(batch_size=[1])

        # Setup specific instance for deterministic test
        # Job 0: M0(2), M1(3), M2(4)
        # Job 1: M0(5), M1(2), M2(1)
        # Job 2: M0(1), M1(1), M2(1)

        # Override data
        td["proc_time"][0] = torch.tensor([
            [2, 3, 4],
            [5, 2, 1],
            [1, 1, 1]
        ], dtype=torch.float)

        # All jobs want Machine 0 first, then M1, then M2 (simplest case)
        # Wait, usually it varies. Let's assume order is 0, 1, 2 for all
        td["machine_order"][0] = torch.tensor([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ])

        # Step 1: Schedule Job 0 on Machine 0
        td["action"] = torch.tensor([0])
        td = env.step(td)["next"]

        # Job 0 Next Op Index: 1
        assert td["next_op_idx"][0, 0] == 1
        # Job 0 Avail Time: 2 (start 0 + dur 2)
        assert td["job_avail_time"][0, 0] == 2
        # Machine 0 Avail Time: 2
        assert td["machine_avail_time"][0, 0] == 2

        # Step 2: Schedule Job 1 on Machine 0
        # Should start at Machine 0 avail time (2), duration 5 => end 7
        td["action"] = torch.tensor([1])
        td = env.step(td)["next"]

        assert td["next_op_idx"][0, 1] == 1
        assert td["job_avail_time"][0, 1] == 7
        assert td["machine_avail_time"][0, 0] == 7 # M0 busy until 7

        # Step 3: Schedule Job 0 on Machine 1
        # J0 avail at 2. M1 avail at 0. Start at max(2, 0) = 2.
        # Duration 3. End at 5.
        td["action"] = torch.tensor([0])
        td = env.step(td)["next"]

        assert td["next_op_idx"][0, 0] == 2
        assert td["job_avail_time"][0, 0] == 5
        assert td["machine_avail_time"][0, 1] == 5

    def test_done_logic(self, env):
        # 1 Job, 1 Machine
        env = JSSPEnv(num_jobs=1, num_machines=1)
        td = env.reset(batch_size=[1])

        # Step 1
        td["action"] = torch.tensor([0])
        td = env.step(td)["next"]

        assert td["finished_jobs"][0, 0] == True
        assert td["done"][0] == True
