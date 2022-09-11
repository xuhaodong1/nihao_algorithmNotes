
        # """
        # :type quality: List[int]
        # :type wage: List[int]
        # :type k: int
        # :rtype: float
        # """


from heapq import heapify, heapreplace
from math import inf

# 优先队列相关练习题
class PriorityQueue:

    # 题目链接：[857. 雇佣 K 名工人的最低成本](https://leetcode.cn/problems/minimum-cost-to-hire-k-workers/)
    def mincostToHireWorkers(self, quality, wage, k) -> float:
        qw = sorted(zip(quality, wage), key=lambda p: p[1] / p[0]) #按照 wage[i] / quality[i] 排序
        h = [-q for q, _ in qw[:k]]
        heapify(h)
        sum_q = -sum(h)
        ans = sum_q * qw[k - 1][1] / qw[k - 1][0]
        for q, w in qw[k:]: # 之前的按照单位 qw[k - 1][1] / qw[k - 1][0] 处理
            if q < -h[0]:
                sum_q += heapreplace(h, -q) + q
                ans = min(ans, sum_q * w / q)
        return ans

p = PriorityQueue()
print(p.mincostToHireWorkers([3,1,10,10,1], [4,8,2,2,7], 3))
