//
//  MonotoneStack.swift
//  LeetCode
//
//  Created by haodong xu on 2022/9/1.
//

import Foundation

/// 单调栈相关练习题
class MonotoneStack {

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
}
