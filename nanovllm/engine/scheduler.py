from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    nano-vllm 的请求调度器
    主要职责:
        1. 管理 waiting 和 running 两个请求队列
        2. 决定本轮执行 prefill 还是 decode
        3. 控制每轮最大请求数和 prefill token budget
        4. 通过 BlockManager 分配、复用和释放 KV Cache Block
        5. 显存不足时执行请求抢占(preemption)
        6. 模型执行后更新 Token、缓存进度和请求状态
    注意：
    Scheduler 不负责执行 Transformer Forward。
    真正的 GPU 模型计算由 ModelRunner 完成。
        
    你提供的这个 nano-vllm 版本采用 Prefill 优先的两阶段调度，一次 schedule() 只返回 Prefill Batch 或 Decode Batch，不会将两者混合到同一次模型执行中。
    """
    def __init__(self, config: Config):
        # 这里存在两种资源限制：序列数量（控制批次中最多有多少条请求）、Prefill Token Budget（控制 Prefill 批次一次最多处理多少 Token） 
        self.max_num_seqs = config.max_num_seqs  # 每次调度最多选择多少条 Sequence
        self.max_num_batched_tokens = config.max_num_batched_tokens     # Prefill 阶段单次调度的 Token Budget
        self.eos = config.eos   # 模型的结束 Token ID
        self.block_size = config.kvcache_block_size  # 每个 KV Cache Block 容纳的 Token 数
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size) # 初始化 KV Cache Block 管理层


        # Waiting 队列：
        # 1. 新加入、尚未完成 Prefill 的请求
        # 2. 正在执行 Chunked Prefill 的请求
        # 3. 被抢占后等待恢复的请求
        self.waiting: deque[Sequence] = deque() 

        # Running 队列：
        # 已被安排完成 Prefill、可参与后续 Decode 的请求。
        # 注意：RUNNING 不代表 GPU 此刻正在执行该请求。        
        self.running: deque[Sequence] = deque() # 已经完成 Prompt 计算，可以继续执行 Decode 的请求。

    def is_finished(self):  
        """
        判断所有请求是否已经完成。
        只有 Waiting 和 Running 都为空，才认为整个请求批次处理结束。
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):   
        """
        将新请求加入 Waiting 队列尾部。不会立即执行模型计算。
        """
        self.waiting.append(seq)

    # ---------- 核心方法：调度 ----------
    # 由此可以看出，这个 Scheduler 的调度策略是：
    # 1. 优先选择 Waiting 队列里的请求执行 Prefill。
    # 2. 如果本轮已经选中了至少一条 Prefill 请求，就直接返回 Prefill Batch。
    # 3. 只有本轮没有选中任何 Prefill 请求，才进入 Decode 调度。
    # 4. 在 Decode 时，如果 KV Block 不够，可以抢占其他 Running 请求。

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度函数。
        返回：
            scheduled_seqs, 本轮选中、准备执行的 Sequence 列表。
            is_prefill:
                True  -> 本轮执行 Prefill
                False -> 本轮执行 Decode
        当前调度策略：
            先尝试 Prefill。
            只要成功选中至少一个 Prefill 请求，就直接返回，不再选择 Decode 请求。

            只有没有选中 Prefill 请求时，才执行 Decode 调度。
        """
        scheduled_seqs = []     # 保存本轮选中的请求。
        num_batched_tokens = 0  # 记录本轮已经安排的 Prefill Token 数量。

        # =====================================================
        # 第一阶段：Prefill 调度
        # =====================================================
        # 循环条件：
        # 1. Waiting 队列中还有请求
        # 2. 本轮选中的请求数没有超过 max_num_seqs
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs: # Waiting 队列中仍然存在请求; 本轮已选择的请求数量没有达到 max_num_seqs
            # 查看 Waiting 队首请求。
            # 此处只读取，不立即从队列移除。
            #
            # 因为后面还要判断：
            # - Token Budget 是否足够
            # - KV Cache Block 是否足够
            # - 是否允许本轮 Chunked Prefill
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens # 计算本轮剩余的 Prefill Token Budget。 
            if remaining == 0:  # 如果本轮 Token Budget 已耗尽，不再继续选择 Prefill 请求。 
                break
            # -------------------------------------------------
            # 1. 判断是否已有 KV Block 映射
            # -------------------------------------------------
            if not seq.block_table: # 这个请求之前没有分配过 KV Block  
                # block_table 为空，说明当前请求尚未持有已分配的物理 KV Block。
                # 可以是：
                # - 新请求
                # - 被抢占后重新进入 Waiting 的请求

                # 检查是否能够分配需要的 KV Block。
                # 同时检查是否存在可复用的前缀缓存。
                # 返回：
                # -1：物理 Block 资源不足，无法分配
                # >=0：连续命中的完整前缀 Block 数量
                num_cached_blocks = self.block_manager.can_allocate(seq)    # 返回值:(-1, 当前可用 Block 不足，无法完成所需分配; 0, 可以分配, 但没有可复用的前缀 Block; 正整数, 可复用的完整前缀 Block 数量) 
                # 如果当前没有足够的 Block，本轮停止 Prefill 调度。
                # 注意：这里是 break，而不是 continue。因此不会跳过队首请求去尝试后面的请求。
                if num_cached_blocks == -1:
                    break

                # 本次需要计算的 Token 数。
                # 如果前缀缓存命中，已经计算过的
                # 完整 Block 无须再次执行 Prefill。
                # 例如：
                # num_tokens=600，block_size=256，
                # num_cached_blocks=2
                # 需要处理：
                # 600 - 2*256 = 88 Tokens
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size   # 这就是 Prefix Caching 对 Prefill 调度产生的实际影响
            else:
                # block_table 不为空，说明请求之前
                # 已经分配过 KV Block。
                # 常见情况是：
                # 长 Prompt 正在执行 Chunked Prefill，其中部分 Token 已经计算完成。
                # 只需要继续计算尚未缓存的 Token。
                num_tokens = seq.num_tokens - seq.num_cached_tokens # 直接用 总 Token 数 - 已缓存 Token 数, 得到尚未完成计算的长度
            
            # -------------------------------------------------
            # 2. 判断是否允许 Chunked Prefill
            # -------------------------------------------------
            # 如果：
            # 1. 当前剩余 Token Budget 不够完成这个请求
            # 2. 本轮已经选择了其他请求
            #
            # 那么本轮不再选择当前请求。
            #
            # 这样做的效果是：
            # 只允许本轮第一个被选中的 Sequence
            # 执行 Chunked Prefill。
            #
            # 后续被选中的 Sequence 必须能够
            # 在剩余预算内完成它们本轮剩余的 Prefill。  
            # # 只允许本轮第一个请求做 chunked prefill          
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            # -------------------------------------------------
            # 3. 实际分配 KV Block
            # -------------------------------------------------
            if not seq.block_table:
                # 如果此前没有 Block Table，
                # 就为当前请求建立物理 Block 映射。
                #
                # allocate() 会：
                # - 复用命中的前缀 Block
                # - 为未命中部分分配新 Block
                # - 更新 seq.block_table
                # - 设置 seq.num_cached_tokens
                #
                # 重要：
                # 当前版本会为整个现有 Sequence
                # 预先准备所需 Block，而不是只为
                # 本轮 Chunked Prefill 片段分配 Block。                
                self.block_manager.allocate(seq, num_cached_blocks)
            # 本轮实际安排的 Token 数量：
            #
            # 取以下二者的较小值：
            # 1. 当前请求尚未计算的 Token 数
            # 2. 本轮剩余 Token Budget
            #
            # 如果预算不足且这是本轮第一个请求，
            # 就只处理一个 Prefill Chunk。                
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens             # 更新本轮累计的 Prefill Token 数量。
            # -------------------------------------------------
            # 4. 判断本轮是否可以完成 Prefill
            # -------------------------------------------------
            # 已计算 Token + 本轮待计算 Token
            # 如果正好等于当前请求的全部 Token，
            # 说明这次模型执行之后将完成 Prefill。
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 提前将请求标记为 RUNNING。
                # 注意：此时还没有真正执行 Forward。只是已经安排好本轮完成 Prefill。
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        # -----------------------------------------------------
        # 5. Prefill 优先
        # -----------------------------------------------------
        # 只要本轮选中了至少一个 Prefill 请求，
        # 就立即返回 Prefill Batch。本轮不会再选择 Running 队列中的 Decode 请求。
        if scheduled_seqs:
            return scheduled_seqs, True

        # =====================================================
        # 第二阶段：Decode 调度
        # =====================================================

        # 只有本轮没有选择任何 Prefill 请求，才会进入下面的 Decode 调度。
        # 循环条件：
        # 1. Running 队列非空
        # 2. 本轮选择的请求数小于 max_num_seqs
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            # 从 Running 队首取出一个请求。与 Prefill 不同，这里会立刻移出队列。
            seq = self.running.popleft()        
            # -------------------------------------------------
            # 1. 检查当前 Token 能否写入 KV Cache
            # -------------------------------------------------

            # Decode 时，每条 Sequence 通常只计算最新生成、但尚未写入 KV Cache 的一个 Token。
            # 如果当前 Block 已满，就需要新物理 Block。
            #
            # can_append() 用来判断当前缓存资源是否足够支持本轮 Decode。            
            
            while not self.block_manager.can_append(seq):   # 这个函数检查本轮 Decode 是否具备足够的 KV Block 资源。
                # 资源不足时优先抢占其他请求，尽可能保留当前从队首取出的请求。
                if self.running:
                    self.preempt(self.running.pop()) # 从 Running 队尾选择一个请求抢占。
                else:
                    self.preempt(seq)   # 如果没有其他可供抢占的 Running 请求，就只能抢占当前请求自身
                    break   # 放弃本轮对该请求的调度
            else:
                # 只有 while 正常退出（没有执行 break）才进入这里, 即 can_append(seq) 最终返回 True，当前请求具有执行所需的缓存资源。
                seq.num_scheduled_tokens = 1    # 常规 Decode 每轮处理一个 Token
                seq.is_prefill = False
                self.block_manager.may_append(seq)  # 如果当前 Token 需要新的 KV Block, 就实际分配，并追加到 block_table。 如果当前 Block 还有空间，则不会新分配 Block
                scheduled_seqs.append(seq)
        
        # 当前设计假设：
        # 必须至少成功选择一个请求执行。
        #
        # 如果既不能 Prefill，也不能 Decode，
        # 这里可能发生 AssertionError。        
        assert scheduled_seqs

        # -----------------------------------------------------
        # 2. 恢复 Running 队列
        # -----------------------------------------------------

        # 前面使用 popleft() 将选中的请求移出了队列。现在需要把它们放回队首。
        # reversed() 配合 extendleft()，
        # 可以保持选中请求原来的顺序。
        # 例如：
        # 原队列：[A, B, C]
        # 选中：[A, B]
        # 剩余：[C]
        # 恢复后：[A, B, C]

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False # 返回本轮 Decode 请求。
    
    def preempt(self, seq: Sequence):
        """
        抢占一个正在运行的请求。

        目的：
        释放这个请求占用的 KV Block，
        让其他请求可以继续执行。

        当前版本采用重计算式恢复思路，
        并没有将整份 KV Cache 换出到 CPU。
        """

        seq.status = SequenceStatus.WAITING # 请求重新进入等待状态。
        seq.is_prefill = True   # 后续恢复时重新使用 Prefill 路径。
        self.block_manager.deallocate(seq)  # 释放当前请求占有的物理 KV Block 引用。但不会删除 seq.token_ids。因此请求的 Prompt 和已经生成的 Token, 仍然保留，可用于恢复时重新计算 KV。
        # Prefix Cache 中的完整 Block 如果仍有缓存记录，后续还可能被复用
        self.waiting.appendleft(seq) # 将被抢占的请求放到 Waiting 队首, 使它在后续 Prefill 调度中优先被考虑。

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        一次 ModelRunner.run() 执行结束后的状态更新。

        参数：
            seqs：
                本轮执行过模型计算的 Sequence。

            token_ids：
                ModelRunner 返回的采样 Token IDs。

            is_prefill：
                本轮是否执行 Prefill。

        核心工作：
        1. 为新计算完成的完整 KV Block 建立哈希记录。
        2. 更新 num_cached_tokens。
        3. 对可以正式生成输出的请求追加 Token。
        4. 检查 EOS 和 max_tokens。
        5. 释放已经结束的请求所占用的缓存资源。
        """
        for seq, token_id in zip(seqs, token_ids):
            # -------------------------------------------------
            # 1. 更新 Prefix Cache
            # -------------------------------------------------

            # 为本轮新计算完成的完整 KV Block, 计算哈希并登记缓存。
            # 注意：
            # hash_blocks() 根据更新前的
            # num_cached_tokens 和 num_scheduled_tokens
            # 判断哪些 Block 刚刚完成计算。
            #
            # 所以这里必须先 hash_blocks，
            # 再更新 num_cached_tokens。            
            self.block_manager.hash_blocks(seq)
            # -------------------------------------------------
            # 2. 更新 KV Cache 计算进度
            # -------------------------------------------------

            # 本轮安排的 Token 已经完成模型 Forward，
            # 对应的 KV 已写入缓存。

            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0        # 本轮调度任务结束，清空调度数量。
            # -------------------------------------------------
            # 3. 判断 Chunked Prefill 是否完成
            # -------------------------------------------------
            # 如果当前正在 Prefill，
            # 并且仍有 Token 尚未计算：
            # 说明只完成了部分 Prompt。
            # 此时不能把模型返回的采样 Token, 当成正式生成结果。             
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue    # 等待后续 Prefill 继续处理剩余 Prompt。
            # -------------------------------------------------
            # 4. 追加正式生成的 Token
            # -------------------------------------------------

            # 如果：
            # - Prefill 已全部完成，或者 当前是 Decode 阶段
            # 那么本轮返回的 Token 可以正式追加。
            # append_token() 只更新 Token 序列，
            # 不会立即计算该新 Token 自身的 KV。

            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
