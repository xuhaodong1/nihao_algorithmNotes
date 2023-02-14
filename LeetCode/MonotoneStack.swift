//
//  MonotoneStack.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/1.
//

import Foundation

/// 单调栈相关练习题

/// 题目链接：[901. 股票价格跨度](https://leetcode.cn/problems/online-stock-span/)
class StockSpanner {
    var stack = [(Int, Int)]()
    init() {}

    func next(_ price: Int) -> Int {
        var cnt = 1
        while !stack.isEmpty && price >= stack.last!.0  {
            cnt += stack.removeLast().1
        }
        stack.append((price, cnt))
        return cnt
    }
}

class MonotoneStack: BaseCode {

    /// 题目链接：[1475. 商品折扣后的最终价格](https://leetcode.cn/problems/final-prices-with-a-special-discount-in-a-shop/)
    func finalPrices(_ prices: [Int]) -> [Int] {
        var stack = [Int](), ans = prices
        for (i, price) in prices.enumerated() {
            while !stack.isEmpty && prices[stack.last!] >= price {
                let last = stack.removeLast()
                ans[last] -= price
            }
            stack.append(i)
        }
        return ans
    }

    /// 题目链接：[907. 子数组的最小值之和](https://leetcode.cn/problems/sum-of-subarray-minimums/)
    /// 相关题目：[828. 统计子串中的唯一字符](https://leetcode.cn/problems/count-unique-characters-of-all-substrings-of-a-given-string/)
    func sumSubarrayMins(_ arr: [Int]) -> Int {
        let n = arr.count, mod = Int(1e9) + 7
        var stack = [Int]()
        var left = [Int](repeating: n, count: n), right = [Int](repeating: -1, count: n)
        var ans = 0
        for i in 0..<n {
            while !stack.isEmpty && arr[i] <= arr[stack.last!] {
                left[stack.removeLast()] = i
            }
            stack.append(i)
        }
        stack.removeAll()
        for i in (0..<n).reversed() {
            while !stack.isEmpty && arr[i] < arr[stack.last!] {
                right[stack.removeLast()] = i
            }
            stack.append(i)
        }
        for i in 0..<n {
            ans += ((((left[i] - i) * (i - right[i])) % mod) * arr[i]) % mod
            ans %= mod
        }
        return ans
    }

    /// 题目链接：[1124. 表现良好的最长时间段](https://leetcode.cn/problems/longest-well-performing-interval/)
    /// 计算前缀和之后的三种做法
    /// 1. 双重循环枚举所有情况
    /// 2. 单调栈列举可能成为左端点的情况
    /// 3. 利用前缀和连续性，以 map 或者 arr 计算
    func longestWPI(_ hours: [Int]) -> Int {
        let n = hours.count
        var ans = 0
        var preSum = [Int](repeating: 0, count: n + 1)
        var queue = [0]
        for (i, hour) in hours.enumerated() {
            preSum[i + 1] = preSum[i] + (hour > 8 ? 1 : -1)
            if preSum[i + 1] < preSum[queue.last!] { queue.append(i + 1) }
        }
        for i in (1...n).reversed() {
            while !queue.isEmpty && preSum[i] > preSum[queue.last!] {
                ans = max(ans, i - queue.removeLast())
            }
        }
        return ans
    }

//    override var excuteable: Bool { return true }

    override func executeTestCode() {
        super.executeTestCode()
//        print(longestWPI([10, 10, 1, 1, 1, 1]))
    }
}
