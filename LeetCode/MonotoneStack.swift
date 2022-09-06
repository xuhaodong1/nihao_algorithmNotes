//
//  MonotoneStack.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/1.
//

import Foundation

/// 单调栈相关练习题
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
                let lastIndex = stack.removeLast()
                left[lastIndex] = i
            }
            stack.append(i)
        }
        stack.removeAll()
        for i in (0..<n).reversed() {
            while !stack.isEmpty && arr[i] < arr[stack.last!] {
                let lastIndex = stack.removeLast()
                right[lastIndex] = i
            }
            stack.append(i)
        }
        for i in 0..<n {
            ans += ((((left[i] - i) * (i - right[i])) % mod) * arr[i]) % mod
            ans %= mod
        }
        return ans
    }

//    override var excuteable: Bool {
//        return true
//    }

    override func executeTestCode() {
        super.executeTestCode()
    }
}
