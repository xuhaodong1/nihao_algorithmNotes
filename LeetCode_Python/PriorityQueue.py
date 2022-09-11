from heapq import heapify, heapreplace
from math import inf

# 优先队列相关练习题
class PriorityQueue:

    # 题目链接：[857. 雇佣 K 名工人的最低成本](https://leetcode.cn/problems/minimum-cost-to-hire-k-workers/)
    def mincostToHireWorkers(self, quality, wage, k) -> float:
        qw = sorted(zip(quality, wage), key=lambda p: p[1] / p[0]) #按照 wage[i] / quality[i] 排序
        h = [-q for q, _ in qw[:k]] # 加负号变成最大堆
        heapify(h)
        sum_q = -sum(h)
        ans = sum_q * qw[k - 1][1] / qw[k - 1][0] # 选 w / q 值最小的 k 名工人组成当前的最优解
        for q, w in qw[k:]:
            if q < -h[0]: # sum_q 可以变小，从而可能得到更优的答案, 而 w / q 是递增的只会变大
                sum_q += heapreplace(h, -q) + q
                ans = min(ans, sum_q * w / q)
        return ans

p = PriorityQueue()
print(p.mincostToHireWorkers([3,1,10,10,1], [4,8,2,2,7], 3))
