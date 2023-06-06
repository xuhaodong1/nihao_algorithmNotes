from heapq import heapify, heappop, heappush, heapreplace
from typing import List

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

    # 题目链接：[1792. 最大平均通过率](https://leetcode.cn/problems/maximum-average-pass-ratio/)
    def maxAverageRatio(self, classes: List[List[int]], extraStudents: int) -> float:
        h = [(a / b - (a + 1) / (b + 1), a, b) for a, b in classes]
        heapify(h)
        for _ in range(extraStudents):
            _, a, b = heappop(h)
            a, b = a + 1, b + 1
            heappush(h, (a / b - (a + 1) / (b + 1), a, b))
        return sum(v[1] / v[2] for v in h) / len(classes)

    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mask = 1 << len(nums)
        ans = 0
        for i in range(1, mask):
            set = {-1}
            isFail = False
            for j in range(n):
                if i & (1 << j):
                    if nums[j] - k in set or nums[j] + k in set:
                        isFail = True
                        break
                    set.add(nums[j])
            if not isFail:
                ans += 1
        return ans
    # func beautifulSubsets(_ nums: [Int], _ k: Int) -> Int {
    #     let n = nums.count
    #     let mask = 1 << n
    #     var ans = 0
    #     var set = Set<Int>()
    #     for i in 1..<mask {
    #         set.removeAll()
    #         var isFail = false
    #         for j in 0..<n where i & (1 << j) > 0 {
    #             if set.contains(nums[j] - k) || set.contains(nums[j] + k) {
    #                 isFail = true
    #                 break
    #             }
    #             set.insert(nums[j])
    #         }
    #         if !isFail {
    #             ans += 1
    #         }
    #     }
    #     return ans
    # }

p = PriorityQueue()
# print(p.maxAverageRatio([[1,2],[3,5],[2,2]], 2))
print(p.beautifulSubsets([13,18,17,11,9,4,2,14,12,7,8,16,5,20,19,10,15,3,6,1],
                               13))
