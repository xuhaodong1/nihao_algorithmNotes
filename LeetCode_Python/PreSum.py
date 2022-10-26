
from ast import List
from cmath import inf
from collections import deque
from itertools import accumulate


class Solution:

    # 前缀和 + 单调队列
    # 题目链接：[862. 和至少为 K 的最短子数组](https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/)
    # 参考 https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/solution/liang-zhang-tu-miao-dong-dan-diao-dui-li-9fvh/
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        ans = inf
        preSum = list(accumulate(nums, initial=0))
        q = deque()
        for i, curr_s in enumerate(preSum):
            while q and curr_s - preSum[q[0]] >= k:
                ans = min(ans, i - q.popleft())
            while q and preSum[q[-1]] >= curr_s:
                q.pop()
            q.append(i)
        return ans if ans < inf else -1