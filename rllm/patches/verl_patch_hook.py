import wrapt

_TARGET = "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd"


def setup():
    @wrapt.when_imported(_TARGET)
    def _patch(mod):
        # Your replacement method, defined here so nothing heavy is imported early.
        def _patched_init_zeromq(self) -> str:
            import getpass
            import os
            import threading

            import zmq
            from filelock import FileLock

            tensor_parallel_size = self.config.tensor_model_parallel_size
            local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
            socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

            user = getpass.getuser()
            with FileLock(f"/tmp/verl_vllm_zmq_{user}.lock"):
                if socket_type == "ipc":
                    pid = os.getpid()
                    address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{user}.ipc"
                else:
                    ip, port = self._get_free_port()
                    address = f"tcp://{ip}:{port}"

                context = zmq.Context()
                self.socket = context.socket(zmq.REP)
                self.socket.bind(address)

            self.loop_thread = threading.Thread(target=self._loop_forever)
            self.loop_thread.start()
            return address

        Cls = getattr(mod, "vLLMAsyncRollout", None)
        if Cls is not None:
            Cls._init_zeromq = _patched_init_zeromq
